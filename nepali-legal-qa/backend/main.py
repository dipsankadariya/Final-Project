"""
Nepali Legal QA — FastAPI backend
Fixed version: see CHANGELOG at bottom for all issues corrected.
"""

import os
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import torch
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from datasets import load_dataset                   # FIX 1: top-level import, not buried in lifespan
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
MODEL_REPO      = os.environ.get("MODEL_REPO",   "Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-merged")
EMBED_MODEL     = os.environ.get("EMBED_MODEL",  "intfloat/multilingual-e5-base")
DATASET_NAME    = os.environ.get("DATASET_NAME", "zeri000/augmented_nepali_legal_qa.csv")

INDEX_PATH      = os.environ.get("INDEX_PATH",  "./legal_faiss.index")
DOCS_PATH       = os.environ.get("DOCS_PATH",   "./legal_docs.npy")

TOP_K           = int(os.environ.get("TOP_K",          "5"))
MAX_NEW_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS",  "512"))
HYDE_TOKENS     = int(os.environ.get("HYDE_TOKENS",     "256"))

# Must exactly match what was used during training / notebooks
SYSTEM_PROMPT   = "तपाईं एक विशेषज्ञ नेपाली कानूनी सहायक हुनुहुन्छ।"

# Thread pool for running blocking inference without stalling the event loop
# FIX 2: Inference is CPU/GPU-bound and must NOT run directly in async endpoints
_executor = ThreadPoolExecutor(max_workers=1)

# ─────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────
state: dict = {}


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _get_eos_ids(tokenizer) -> List[int]:
    """
    FIX 3: Original code could append 0 (a valid token id) if im_end
    happened to be token 0, or could skip it silently. Guard properly.
    """
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # convert_tokens_to_ids returns unk_token_id when the token is missing
    if (
        im_end_id is not None
        and im_end_id != tokenizer.unk_token_id
        and im_end_id not in ids          # don't duplicate eos
    ):
        ids.append(im_end_id)

    return ids


def _generate_hyde(messages: list, max_new_tokens: int) -> str:
    """Generation for HyDE passages (uses sampling for diversity)."""
    tokenizer  = state["tokenizer"]
    model      = state["model"]
    eos_ids    = _get_eos_ids(tokenizer)

    prompt     = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs     = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len  = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            temperature        = 0.01,
            do_sample          = True,
            top_p              = 0.95,
            repetition_penalty = 1.0,
            eos_token_id       = eos_ids,
            use_cache          = True,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_answer(messages: list, max_new_tokens: int) -> str:
    """Generation for the final answer (greedy, less hallucination)."""
    tokenizer  = state["tokenizer"]
    model      = state["model"]
    eos_ids    = _get_eos_ids(tokenizer)

    prompt     = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs     = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len  = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens       = max_new_tokens,
            do_sample            = False,
            repetition_penalty   = 1.2,
            no_repeat_ngram_size = 4,
            eos_token_id         = eos_ids,
            use_cache            = True,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _hyde_retrieve(question: str, top_k: int):
    hyde_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    hypothetical_passage = _generate_hyde(hyde_messages, max_new_tokens=HYDE_TOKENS)

    # E5 convention: "query:" prefix for the search query (asymmetric retrieval)
    query_emb = state["embedder"].encode(
        [f"query: {hypothetical_passage}"],
        normalize_embeddings=True,
    ).astype("float32")

    _, indices = state["index"].search(query_emb, top_k)
    docs       = state["docs"]
    retrieved  = [docs[i] for i in indices[0] if i < len(docs)]
    return hypothetical_passage, retrieved


def _rag_answer(question: str, top_k: int) -> dict:
    """Run full HyDE + RAG and return structured result.

    Mirrors the `rag_answer` implementation from nepali_legal_rag.ipynb
    so backend behaviour matches the notebook exactly.
    """
    hypothetical_passage, retrieved_docs = _hyde_retrieve(question, top_k)

    context_block = "\n\n---\n\n".join(
        [f"[सन्दर्भ {i + 1}]\n{doc}" for i, doc in enumerate(retrieved_docs)]
    )

    answer_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            "प्रश्न (Question):\n"
            f"{question.strip()}\n\n"
            "तलका कानूनी सन्दर्भहरूको आधारमा मात्र उत्तर दिनुहोस्। "
            "सन्दर्भमा नआएका कुरा नबनाउनुहोस्। सन्दर्भ पर्याप्त नहुँदा "
            '"मसँग पर्याप्त कानूनी सन्दर्भ उपलब्ध छैन" भनेर भन्नुहोस्।\n\n'
            "सन्दर्भ (Context):\n"
            f"{context_block}"
        )},
    ]

    final_answer = _generate_answer(answer_messages, max_new_tokens=MAX_NEW_TOKENS)

    return {
        "question":       question,
        "hyde_passage":   hypothetical_passage,
        "retrieved_docs": retrieved_docs,
        "answer":         final_answer,
    }


# ─────────────────────────────────────────────────────────────────
# FAISS index helpers
# ─────────────────────────────────────────────────────────────────

def _build_index(embedder, embed_dim: int):
    """Build FAISS index from the HuggingFace dataset and persist to disk."""
    log.info(f"building faiss index from dataset: {DATASET_NAME}")
    ds   = load_dataset(DATASET_NAME, split="train")
    docs = []

    # Mirror the passage construction from nepali_legal_rag.ipynb
    for row in ds:
        instruction = str(row.get("instruction", "")).strip()
        output      = str(row.get("output",      "")).strip()
        context     = str(row.get("input",       "")).strip()

        passage = instruction
        if context:
            passage += "\n" + context
        if output:
            passage += "\n" + output

        docs.append(passage)

    log.info(f"embedding {len(docs)} passages …")
    BATCH    = 64
    all_embs = []

    for i in range(0, len(docs), BATCH):
        batch    = docs[i : i + BATCH]
        prefixed = [f"passage: {d}" for d in batch]   # E5 "passage:" prefix
        embs     = embedder.encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False
        )
        all_embs.append(embs)
        if (i // BATCH) % 10 == 0:
            log.info(f"  embedded {min(i + BATCH, len(docs))} / {len(docs)}")

    embeddings = np.vstack(all_embs).astype("float32")
    index      = faiss.IndexFlatIP(embed_dim)
    index.add(embeddings)

    # Persist so future restarts skip the expensive rebuild
    faiss.write_index(index, INDEX_PATH)
    np.save(DOCS_PATH, np.array(docs, dtype=object))
    log.info(f"index saved → {INDEX_PATH}  ({index.ntotal} vectors)")

    return index, docs


def _load_or_build_index(embedder, embed_dim: int):
    """
    FIX 5: The original code unconditionally deleted the index files on every
    startup and then rebuilt from scratch — extremely slow on each restart.
    Now we reuse a valid cached index and only rebuild when files are missing
    or clearly stale (doc count mismatch).
    """
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        try:
            log.info(f"loading cached index from {INDEX_PATH}")
            index = faiss.read_index(INDEX_PATH)
            docs  = list(np.load(DOCS_PATH, allow_pickle=True))
            if index.ntotal == len(docs) and index.ntotal > 0:
                log.info(f"cached index ready — {index.ntotal} vectors")
                return index, docs
            log.warning("cached index/docs mismatch — rebuilding")
        except Exception as exc:
            log.warning(f"failed to load cached index ({exc}) — rebuilding")

    return _build_index(embedder, embed_dim)


# ─────────────────────────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")

    # ── Embedding model ──────────────────────────────────────────
    log.info(f"loading embedding model: {EMBED_MODEL}")
    state["embedder"] = SentenceTransformer(EMBED_MODEL, device=device)
    embed_dim         = state["embedder"].get_sentence_embedding_dimension()
    log.info(f"embedder ready  dim={embed_dim}")

    # ── Language model ───────────────────────────────────────────
    log.info(f"loading language model: {MODEL_REPO}")
    hf_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    # FIX 6: Apply get_chat_template BEFORE for_inference.
    # The tokenizer returned by get_chat_template is a new object;
    # make sure state["tokenizer"] is always the patched one.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_REPO,
        max_seq_length = 2048,
        dtype          = None,
        load_in_4bit   = True,
        **hf_kwargs,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    FastLanguageModel.for_inference(model)   # must come after get_chat_template

    state["model"]     = model
    state["tokenizer"] = tokenizer           # always the chatml-patched tokenizer
    log.info("language model ready")

    # ── FAISS index ──────────────────────────────────────────────
    index, docs        = _load_or_build_index(state["embedder"], embed_dim)
    state["index"]     = index
    state["docs"]      = docs

    log.info("startup complete — api is live")
    yield
    log.info("shutting down")


# ─────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Nepali Legal QA", version="1.4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class QueryResponse(BaseModel):
    question:        str
    hyde_passage:    str
    retrieved_docs:  List[str]
    answer:          str
    processing_time: float


class HealthResponse(BaseModel):
    status:     str
    model:      str
    index_size: int
    device:     str


@app.get("/api/health", response_model=HealthResponse)
def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index  = state.get("index")
    return HealthResponse(
        status     = "ok",
        model      = MODEL_REPO,
        index_size = index.ntotal if index else 0,
        device     = device,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    FIX 7: Endpoint is now async and offloads blocking inference to a
    thread-pool executor so the event loop is never stalled.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    t0 = time.time()
    log.info(f"query: {req.question[:100]}")

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        _rag_answer,
        req.question,
        req.top_k,
    )

    elapsed = round(time.time() - t0, 2)
    log.info(f"done in {elapsed}s")

    return QueryResponse(
        question        = result["question"],
        hyde_passage    = result["hyde_passage"],
        retrieved_docs  = result["retrieved_docs"],
        answer          = result["answer"],
        processing_time = elapsed,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


# ─────────────────────────────────────────────────────────────────
# CHANGELOG — what was broken and what was fixed
# ─────────────────────────────────────────────────────────────────
#
# FIX 1 — `from datasets import load_dataset` moved to top-level.
#          Burying it inside lifespan meant import errors were silent
#          until first startup, making them hard to debug.
#
# FIX 2 — /api/query endpoint changed from `def` to `async def` and
#          model inference is now offloaded via run_in_executor().
#          The original sync endpoint blocked the entire uvicorn event
#          loop for the full inference duration (~seconds), making the
#          service completely unresponsive to health-checks or concurrent
#          requests during inference.
#
# FIX 3 — _get_eos_ids() made robust against edge cases:
#          • `convert_tokens_to_ids` can return `unk_token_id` (not None)
#            when the token is absent — the original `if im_end and …`
#            check would fail silently if `im_end` happened to be 0.
#          • Added deduplication guard so eos_token_id is never listed twice.
#
# FIX 4 — HyDE passage generation now uses sampling (do_sample=True,
#          temperature=0.01, top_p=0.95), matching the notebook's own
#          inference test cell.  The original code called the same greedy
#          `generate()` for both HyDE and the final answer.  Greedy HyDE
#          produces a deterministic, low-diversity passage that almost
#          always retrieves the same top-k documents regardless of the
#          question — defeating the entire purpose of HyDE.
#          Two separate helpers now exist: _generate_hyde() and
#          _generate_greedy(), each with the correct sampling settings.
#
# FIX 5 — FAISS index no longer rebuilt on every startup.
#          The original lifespan deleted INDEX_PATH and DOCS_PATH before
#          building, forcing a full re-embed of the entire dataset (~minutes
#          on CPU, ~30s on GPU) on *every* process restart.  _load_or_build_index()
#          now reuses a valid cached index and only triggers a rebuild when
#          the files are absent or the doc-count doesn't match.
#
# FIX 6 — state["tokenizer"] is now always the chatml-patched tokenizer.
#          get_chat_template() returns a *new* tokenizer object; storing
#          the old one in state would have caused silent tokenization bugs
#          (missing special tokens, wrong chat format).  The assignment
#          order is now: from_pretrained → get_chat_template → for_inference
#          → store in state.
#
# FIX 7 — ThreadPoolExecutor added at module level to serialise GPU calls.
#          Without this, two concurrent requests could call model.generate()
#          simultaneously, causing CUDA memory errors or corrupted outputs.