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
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
MODEL_REPO      = os.environ.get("MODEL_REPO",      "Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-merged")
BASE_MODEL_REPO = os.environ.get("BASE_MODEL_REPO", "unsloth/Qwen2.5-1.5B-Instruct")
EMBED_MODEL     = os.environ.get("EMBED_MODEL",     "intfloat/multilingual-e5-base")
DATASET_NAME    = os.environ.get("DATASET_NAME",    "zeri000/augmented_nepali_legal_qa.csv")

INDEX_PATH      = os.environ.get("INDEX_PATH",  "./legal_faiss.index")
DOCS_PATH       = os.environ.get("DOCS_PATH",   "./legal_docs.npy")

TOP_K           = int(os.environ.get("TOP_K",          "5"))
MAX_NEW_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS",  "512"))
HYDE_TOKENS     = int(os.environ.get("HYDE_TOKENS",     "256"))

# ── Retrieval tuning (NEW — only change vs original) ─────────────
HYDE_WEIGHT     = float(os.environ.get("HYDE_WEIGHT",   "0.6"))
QUERY_WEIGHT    = float(os.environ.get("QUERY_WEIGHT",  "0.4"))
MIN_SIMILARITY  = float(os.environ.get("MIN_SIMILARITY","0.35"))

# ── Same system prompt as original (model was fine-tuned with this) ──
SYSTEM_PROMPT   = "तपाईं एक विशेषज्ञ नेपाली कानूनी सहायक हुनुहुन्छ।"

_executor = ThreadPoolExecutor(max_workers=1)
state: dict = {}


def _get_model_and_tokenizer(kind: str = "finetuned"):
    """Return (model, tokenizer) pair for either base or fine-tuned SLM.

    kind: "base" | "finetuned" (default)
    """
    if kind == "base":
        return state["base_model"], state["base_tokenizer"]
    return state["model"], state["tokenizer"]

def _get_eos_ids(tokenizer) -> List[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if (
        im_end_id is not None
        and im_end_id != tokenizer.unk_token_id
        and im_end_id not in ids
    ):
        ids.append(im_end_id)

    return ids


def _generate_with_slm(messages: list, max_new_tokens: int, kind: str = "finetuned") -> str:
    """Generate text — EXACT same params as original working version."""
    model, tokenizer = _get_model_and_tokenizer(kind)
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
            repetition_penalty = 1.2,
            eos_token_id       = eos_ids,
            use_cache          = True,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _hyde_retrieve(question: str, top_k: int, kind: str = "finetuned"):
    """
    IMPROVED: Hybrid HyDE retrieval.
    Blends original question embedding (40%) with HyDE passage embedding (60%)
    so retrieval works even when the HyDE passage is poor.
    """
    hyde_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    hypothetical_passage = _generate_with_slm(
        hyde_messages,
        max_new_tokens=HYDE_TOKENS,
        kind=kind,
    )

    # NEW: Embed both original question AND HyDE passage
    embedder = state["embedder"]

    query_emb = embedder.encode(
        [f"query: {question.strip()}"],
        normalize_embeddings=True,
    ).astype("float32")

    hyde_emb = embedder.encode(
        [f"query: {hypothetical_passage}"],
        normalize_embeddings=True,
    ).astype("float32")

    # NEW: Weighted average for robust retrieval
    combined_emb = (QUERY_WEIGHT * query_emb + HYDE_WEIGHT * hyde_emb)
    norm = np.linalg.norm(combined_emb, axis=1, keepdims=True)
    combined_emb = (combined_emb / norm).astype("float32")

    # Search with blended embedding
    scores, indices = state["index"].search(combined_emb, top_k)

    # NEW: Filter by similarity score
    docs       = state["docs"]
    retrieved  = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(docs) and score >= MIN_SIMILARITY:
            retrieved.append(docs[idx])

    # Guarantee at least 1 result
    if not retrieved and indices[0][0] < len(docs):
        retrieved.append(docs[indices[0][0]])

    return hypothetical_passage, retrieved


def _rag_answer(question: str, top_k: int, kind: str = "finetuned") -> dict:
    """Run full HyDE + RAG — same prompt format as original."""
    hypothetical_passage, retrieved_docs = _hyde_retrieve(question, top_k, kind=kind)

    # SAME context format as original
    context_block = "\n\n---\n\n".join(
        [f"[सन्दर्भ {i + 1}]\n{doc}" for i, doc in enumerate(retrieved_docs)]
    )

    # SAME prompt format as original
    answer_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"{question.strip()}"
            "\n\nतलका कानूनी सन्दर्भहरूको आधारमा विस्तृत उत्तर दिनुहोस्:\n\n"
            f"{context_block}"
        )},
    ]

    final_answer = _generate_with_slm(
        answer_messages,
        max_new_tokens=MAX_NEW_TOKENS,
        kind=kind,
    )

    return {
        "question":       question,
        "hyde_passage":   hypothetical_passage,
        "retrieved_docs": retrieved_docs,
        "answer":         final_answer,
    }

def _build_index(embedder, embed_dim: int):
    log.info(f"building faiss index from dataset: {DATASET_NAME}")
    ds   = load_dataset(DATASET_NAME, split="train")
    docs = []

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
        prefixed = [f"passage: {d}" for d in batch]
        embs     = embedder.encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False
        )
        all_embs.append(embs)
        if (i // BATCH) % 10 == 0:
            log.info(f"  embedded {min(i + BATCH, len(docs))} / {len(docs)}")

    embeddings = np.vstack(all_embs).astype("float32")
    index      = faiss.IndexFlatIP(embed_dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    np.save(DOCS_PATH, np.array(docs, dtype=object))
    log.info(f"index saved → {INDEX_PATH}  ({index.ntotal} vectors)")

    return index, docs


def _load_or_build_index(embedder, embed_dim: int):
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")

    log.info(f"loading embedding model: {EMBED_MODEL}")
    state["embedder"] = SentenceTransformer(EMBED_MODEL, device=device)
    embed_dim         = state["embedder"].get_sentence_embedding_dimension()
    log.info(f"embedder ready  dim={embed_dim}")

    log.info(f"loading fine-tuned language model: {MODEL_REPO}")
    hf_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_REPO,
        max_seq_length = 2048,
        dtype          = None,
        load_in_4bit   = True,
        **hf_kwargs,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    FastLanguageModel.for_inference(model)

    state["model"]     = model
    state["tokenizer"] = tokenizer
    log.info("fine-tuned language model ready")

    # Optional: separate base model for comparison mode
    if BASE_MODEL_REPO:
        try:
            log.info(f"loading base language model: {BASE_MODEL_REPO}")
            base_model, base_tokenizer = FastLanguageModel.from_pretrained(
                model_name     = BASE_MODEL_REPO,
                max_seq_length = 2048,
                dtype          = None,
                load_in_4bit   = True,
                **hf_kwargs,
            )
            base_tokenizer = get_chat_template(base_tokenizer, chat_template="chatml")
            FastLanguageModel.for_inference(base_model)

            state["base_model"]     = base_model
            state["base_tokenizer"] = base_tokenizer
            log.info("base language model ready")
        except Exception as exc:
            log.warning(f"failed to load base model ({exc}) — comparison mode base endpoint will be unavailable")

    index, docs        = _load_or_build_index(state["embedder"], embed_dim)
    state["index"]     = index
    state["docs"]      = docs

    log.info("startup complete — api is live")
    yield
    log.info("shutting down")


app = FastAPI(title="Nepali Legal QA", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Allow any origin; we don't use cookies, so credentials can be disabled.
    allow_origins=["*"],
    allow_credentials=False,
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
    base_ok = "base_model" in state and "base_tokenizer" in state
    return HealthResponse(
        status     = "ok",
        model      = MODEL_REPO,
        index_size = index.ntotal if index else 0,
        device     = device,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
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
        "finetuned",
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


@app.post("/api/query_base", response_model=QueryResponse)
async def query_base(req: QueryRequest):
    """Same HyDE + RAG pipeline, but using the *base* SLM.

    This is used only for comparison mode in the UI.
    """
    if "base_model" not in state or "base_tokenizer" not in state:
        raise HTTPException(status_code=503, detail="Base model not loaded on server")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    t0 = time.time()
    log.info(f"[base] query: {req.question[:100]}")

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        _rag_answer,
        req.question,
        req.top_k,
        "base",
    )

    elapsed = round(time.time() - t0, 2)
    log.info(f"[base] done in {elapsed}s")

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