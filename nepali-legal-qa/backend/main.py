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
MODEL_REPO      = os.environ.get("MODEL_REPO",   "Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-merged")
DEFAULT_EMBED_MODEL  = "ritesh-07/nepali-legal-e5-finetuned"
FALLBACK_EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_MODEL          = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)
DATASET_NAME         = os.environ.get("DATASET_NAME", "chhatramani/Nepali_Legal_QA")

INDEX_PATH      = os.environ.get("INDEX_PATH",  "./legal_faiss.index")
DOCS_PATH       = os.environ.get("DOCS_PATH",   "./legal_docs.npy")

TOP_K           = int(os.environ.get("TOP_K",          "5"))
MAX_NEW_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS",  "512"))
HYDE_TOKENS     = int(os.environ.get("HYDE_TOKENS",     "256"))
MIN_SIM         = float(os.environ.get("MIN_SIM",       "0.25"))

SYSTEM_PROMPT   = "तपाईं एक विशेषज्ञ नेपाली कानूनी सहायक हुनुहुन्छ।"

_executor = ThreadPoolExecutor(max_workers=1)
state: dict = {}

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


def _generate_with_slm(messages: list, max_new_tokens: int, for_answer: bool = False) -> str:

    tokenizer  = state["tokenizer"]
    model      = state["model"]
    eos_ids    = _get_eos_ids(tokenizer)

    prompt     = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs     = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len  = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens     = max_new_tokens,
        temperature        = 0.01,
        do_sample          = True,
        top_p              = 0.95,
        repetition_penalty = 1.2,
        eos_token_id       = eos_ids,
        use_cache          = True,
    )

    if for_answer:
        gen_kwargs["no_repeat_ngram_size"] = 4
        gen_kwargs["repetition_penalty"]   = 1.25

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _hyde_retrieve(question: str, top_k: int):
    hyde_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    hypothetical_passage = _generate_with_slm(
        hyde_messages,
        max_new_tokens=HYDE_TOKENS,
        for_answer=False,
    )

    query_text = question.strip()
    query_emb = state["embedder"].encode(
        [f"query: {query_text}"],
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = state["index"].search(query_emb, top_k)
    docs            = state["docs"]

    retrieved = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(docs) and score >= MIN_SIM:
            retrieved.append(docs[idx])

    if not retrieved and len(docs) > 0 and indices[0][0] < len(docs):
        retrieved.append(docs[indices[0][0]])

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
            f"{question.strip()}"
            "\n\nतलका कानूनी सन्दर्भहरूको आधारमा विस्तृत उत्तर दिनुहोस्:\n\n"
            f"{context_block}"
            "\n\nयदि माथि दिइएका सन्दर्भहरूमा आवश्यक जानकारी नपाएमा 'उपलब्ध सन्दर्भमा छैन' भनेर स्पष्ट रूपमा जवाफ दिनुहोस्।"
        )},
    ]

    final_answer = _generate_with_slm(
        answer_messages,
        max_new_tokens=MAX_NEW_TOKENS,
        for_answer=True,
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

        parts = []
        if output:
            parts.append(output)
        if context:
            parts.append(context)
        if instruction:
            parts.append("प्रश्न: " + instruction)

        passage = "\n".join(parts).strip()
        if not passage:
            continue

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
    try:
        embedder = SentenceTransformer(EMBED_MODEL, device=device)
    except Exception as exc:
        log.warning(
            f"failed to load embedder '{EMBED_MODEL}' ({exc}), falling back to {FALLBACK_EMBED_MODEL}"
        )
        embedder = SentenceTransformer(FALLBACK_EMBED_MODEL, device=device)

    state["embedder"] = embedder
    embed_dim         = embedder.get_sentence_embedding_dimension()
    log.info(f"embedder ready  dim={embed_dim}")

    log.info(f"loading language model: {MODEL_REPO}")
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
    log.info("language model ready")

    index, docs        = _load_or_build_index(state["embedder"], embed_dim)
    state["index"]     = index
    state["docs"]      = docs

    log.info("startup complete — api is live")
    yield
    log.info("shutting down")


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