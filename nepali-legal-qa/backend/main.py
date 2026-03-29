import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import torch
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
MODEL_REPO     = os.environ.get("MODEL_REPO",   "Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-merged")
EMBED_MODEL    = os.environ.get("EMBED_MODEL",  "intfloat/multilingual-e5-base")
DATASET_NAME   = os.environ.get("DATASET_NAME", "chhatramani/Nepali_Legal_QA")

INDEX_PATH     = os.environ.get("INDEX_PATH",   "./legal_faiss.index")
DOCS_PATH      = os.environ.get("DOCS_PATH",    "./legal_docs.npy")

TOP_K          = int(os.environ.get("TOP_K",          "5"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS",  "512"))
HYDE_TOKENS    = int(os.environ.get("HYDE_TOKENS",     "200"))

# Must exactly match what was used during training
SYSTEM_PROMPT  = "तपाईं एक विशेषज्ञ नेपाली कानूनी सहायक हुनुहुन्छ।"


# ─────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────
state: dict = {}


def get_eos_ids(tokenizer):
    ids    = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end and im_end != tokenizer.unk_token_id:
        ids.append(im_end)
    return ids


def generate(messages: list, max_new_tokens: int) -> str:
    tokenizer = state["tokenizer"]
    model     = state["model"]

    prompt    = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens       = max_new_tokens,
            do_sample            = False,  # greedy decoding — most stable for small models
            repetition_penalty   = 1.3,    # higher value stops repetition loops
            no_repeat_ngram_size = 4,      # blocks any 4-word phrase from repeating
            eos_token_id         = get_eos_ids(tokenizer),
            use_cache            = True,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def hyde_retrieve(question: str, top_k: int):
    hyde_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    hypothetical_passage = generate(hyde_messages, max_new_tokens=HYDE_TOKENS)

    query_emb = state["embedder"].encode(
        [f"query: {hypothetical_passage}"],
        normalize_embeddings=True,
    ).astype("float32")

    _, indices = state["index"].search(query_emb, top_k)
    retrieved  = [state["docs"][i] for i in indices[0] if i < len(state["docs"])]

    return hypothetical_passage, retrieved


def rag_answer(question: str, top_k: int) -> dict:
    hypothetical_passage, retrieved_docs = hyde_retrieve(question, top_k)

    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(f"[सन्दर्भ {i + 1}]\n{doc}")
    context_block = "\n\n".join(context_parts)

    user_message = (
        f"{question.strip()}\n\n"
        f"सन्दर्भ (Context):\n{context_block}"
    )

    answer_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    final_answer = generate(answer_messages, max_new_tokens=MAX_NEW_TOKENS)

    return {
        "question":       question,
        "hyde_passage":   hypothetical_passage,
        "retrieved_docs": retrieved_docs,
        "answer":         final_answer,
    }


# ─────────────────────────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")

    # Always delete old index files so they rebuild from the correct dataset.
    # Colab caches files on disk between runs which caused wrong results before.
    for old_file in [INDEX_PATH, DOCS_PATH]:
        if os.path.exists(old_file):
            os.remove(old_file)
            log.info(f"deleted old index file: {old_file}")

    # Load embedding model
    log.info(f"loading embedding model: {EMBED_MODEL}")
    state["embedder"] = SentenceTransformer(EMBED_MODEL, device=device)
    embed_dim = state["embedder"].get_sentence_embedding_dimension()
    log.info(f"embedder ready  dim={embed_dim}")

    # Load language model
    log.info(f"loading language model: {MODEL_REPO}")
    hf_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_REPO, **hf_kwargs)

    if device == "cuda":
        state["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.float16,
            device_map="auto",
            **hf_kwargs,
        )
    else:
        state["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            **hf_kwargs,
        )

    state["model"].eval()
    log.info("language model ready")

    # Build FAISS index from the correct dataset
    log.info(f"building faiss index from: {DATASET_NAME}")
    from datasets import load_dataset

    ds   = load_dataset(DATASET_NAME, split="train")
    docs = []

    for row in ds:
        instruction = str(row.get("instruction", "")).strip()
        output      = str(row.get("output",      "")).strip()
        source      = str(row.get("source",      "")).strip()

        passage = ""
        if source and source.lower() != "nan":
            passage += f"स्रोत: {source}\n"
        passage += instruction
        if output and output.lower() != "nan":
            passage += "\n" + output

        docs.append(passage)

    log.info(f"embedding {len(docs)} passages — takes a few minutes")

    BATCH    = 64
    all_embs = []

    for i in range(0, len(docs), BATCH):
        batch    = docs[i : i + BATCH]
        prefixed = [f"passage: {d}" for d in batch]
        embs     = state["embedder"].encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False
        )
        all_embs.append(embs)
        if i % (BATCH * 10) == 0:
            log.info(f"  embedded {min(i + BATCH, len(docs))} / {len(docs)}")

    embeddings     = np.vstack(all_embs).astype("float32")
    state["index"] = faiss.IndexFlatIP(embed_dim)
    state["index"].add(embeddings)
    state["docs"]  = docs

    faiss.write_index(state["index"], INDEX_PATH)
    np.save(DOCS_PATH, np.array(docs, dtype=object))
    log.info(f"index ready — {state['index'].ntotal} vectors")

    log.info("startup complete — api is live")
    yield
    log.info("shutting down")


# ─────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Nepali Legal QA", version="1.2.0", lifespan=lifespan)

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
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    t0 = time.time()
    log.info(f"query: {req.question[:100]}")

    result  = rag_answer(req.question, top_k=req.top_k)
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