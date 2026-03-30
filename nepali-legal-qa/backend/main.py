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
EMBED_MODEL     = os.environ.get("EMBED_MODEL",  "intfloat/multilingual-e5-base")
DATASET_NAME    = os.environ.get("DATASET_NAME", "zeri000/augmented_nepali_legal_qa.csv")

INDEX_PATH      = os.environ.get("INDEX_PATH",  "./legal_faiss.index")
DOCS_PATH       = os.environ.get("DOCS_PATH",   "./legal_docs.npy")

TOP_K           = int(os.environ.get("TOP_K",          "7"))
MAX_NEW_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS",  "768"))
HYDE_TOKENS     = int(os.environ.get("HYDE_TOKENS",     "192"))

# ── Retrieval tuning ──────────────────────────────────────────────
HYDE_WEIGHT     = float(os.environ.get("HYDE_WEIGHT",   "0.6"))   # weight for HyDE embedding
QUERY_WEIGHT    = float(os.environ.get("QUERY_WEIGHT",  "0.4"))   # weight for original question embedding
MIN_SIMILARITY  = float(os.environ.get("MIN_SIMILARITY","0.40"))  # drop passages below this score

# ── Enhanced system prompt ────────────────────────────────────────
# Bilingual so the model fully understands the instructions.
# Tells the model HOW to answer, not just WHAT it is.
SYSTEM_PROMPT = (
    "तपाईं एक विशेषज्ञ नेपाली कानूनी सहायक हुनुहुन्छ। "
    "तपाईंले प्रयोगकर्ताको प्रश्नको उत्तर दिनुपर्छ।\n\n"
    "नियमहरू:\n"
    "1. उत्तर सधैं प्रदान गरिएको सन्दर्भ (context) मा आधारित हुनुपर्छ।\n"
    "2. सन्दर्भमा नभएको कुरा नथप्नुहोस् — तथ्यमा मात्र आधारित रहनुहोस्।\n"
    "3. सम्बन्धित ऐन, धारा, नियम वा दफाको नाम उल्लेख गर्नुहोस्।\n"
    "4. उत्तर स्पष्ट, विस्तृत र क्रमबद्ध (point-wise) दिनुहोस्।\n"
    "5. यदि सन्दर्भमा पर्याप्त जानकारी छैन भने स्पष्ट रूपमा भन्नुहोस्।\n"
    "6. नेपालीमा उत्तर दिनुहोस्।"
)

# ── Prompt used only for HyDE passage generation ─────────────────
# This is intentionally simpler — we just need a rough passage.
HYDE_SYSTEM_PROMPT = (
    "तपाईं एक नेपाली कानूनी विशेषज्ञ हुनुहुन्छ। "
    "प्रयोगकर्ताको प्रश्नको सम्भावित उत्तर एक छोटो अनुच्छेदमा लेख्नुहोस्। "
    "सम्बन्धित ऐन वा नियमको नाम समावेश गर्नुहोस्।"
)

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


def _generate_with_slm(messages: list, max_new_tokens: int) -> str:
    """Generate text using the finetuned SLM with greedy decoding."""
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
            do_sample          = False,        # greedy — factually more accurate
            repetition_penalty = 1.15,         # mild penalty to prevent loops
            eos_token_id       = eos_ids,
            use_cache          = True,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _hyde_retrieve(question: str, top_k: int):
    """
    Hybrid HyDE retrieval:
    1. Generate a hypothetical passage using the SLM
    2. Embed BOTH the original question AND the hypothetical passage
    3. Use a weighted average of both embeddings for FAISS search
    4. Filter results by minimum similarity score
    """
    # Step 1: Generate HyDE passage with the simpler prompt
    hyde_messages = [
        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    hypothetical_passage = _generate_with_slm(hyde_messages, max_new_tokens=HYDE_TOKENS)
    log.info(f"hyde passage ({len(hypothetical_passage)} chars): {hypothetical_passage[:120]}...")

    # Step 2: Embed both the original question AND the HyDE passage
    embedder = state["embedder"]

    query_emb = embedder.encode(
        [f"query: {question.strip()}"],
        normalize_embeddings=True,
    ).astype("float32")

    hyde_emb = embedder.encode(
        [f"query: {hypothetical_passage}"],
        normalize_embeddings=True,
    ).astype("float32")

    # Step 3: Weighted average — blend original question with HyDE passage
    # This makes retrieval robust even when HyDE output is poor
    combined_emb = (QUERY_WEIGHT * query_emb + HYDE_WEIGHT * hyde_emb)
    # Re-normalize after averaging
    norm = np.linalg.norm(combined_emb, axis=1, keepdims=True)
    combined_emb = (combined_emb / norm).astype("float32")

    # Step 4: Search FAISS with the blended embedding
    scores, indices = state["index"].search(combined_emb, top_k)

    # Step 5: Filter by minimum similarity score
    docs = state["docs"]
    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(docs) and score >= MIN_SIMILARITY:
            retrieved.append(docs[idx])
            log.info(f"  doc {idx}: score={score:.4f} (accepted)")
        elif idx < len(docs):
            log.info(f"  doc {idx}: score={score:.4f} (rejected, below {MIN_SIMILARITY})")

    # Guarantee at least 1 result even if scores are low
    if not retrieved and indices[0][0] < len(docs):
        retrieved.append(docs[indices[0][0]])
        log.warning(f"  forced top-1 doc (score={scores[0][0]:.4f}) since all below threshold")

    return hypothetical_passage, retrieved


def _rag_answer(question: str, top_k: int) -> dict:
    """Run full HyDE + RAG and return structured result.

    Improvements over original:
    - Hybrid query-HyDE retrieval for robust document matching
    - Structured RAG prompt that forces grounded, cited answers
    - Similarity filtering to exclude irrelevant passages
    """
    hypothetical_passage, retrieved_docs = _hyde_retrieve(question, top_k)

    # Build a clearly structured context block
    context_block = "\n\n---\n\n".join(
        [f"[सन्दर्भ {i + 1}]\n{doc}" for i, doc in enumerate(retrieved_docs)]
    )

    # ── Improved RAG answer prompt ────────────────────────────────
    # Explicit grounding instructions + structured output request
    user_prompt = (
        f"प्रश्न: {question.strip()}\n\n"
        f"तल {len(retrieved_docs)} वटा कानूनी सन्दर्भहरू दिइएका छन्। "
        "यी सन्दर्भहरूमा आधारित भएर मात्र उत्तर दिनुहोस्।\n\n"
        "निर्देशनहरू:\n"
        "- सन्दर्भमा उल्लेख भएका ऐन, धारा, दफा वा नियमको नाम स्पष्ट रूपमा उल्लेख गर्नुहोस्।\n"
        "- उत्तर क्रमबद्ध (point-wise) र विस्तृत दिनुहोस्।\n"
        "- सन्दर्भमा नभएको कुरा नथप्नुहोस्।\n"
        "- यदि सन्दर्भमा पर्याप्त जानकारी छैन भने यो कुरा स्पष्ट पार्नुहोस्।\n\n"
        f"सन्दर्भहरू:\n\n{context_block}\n\n"
        "माथिका सन्दर्भहरूमा आधारित उत्तर:"
    )

    answer_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    final_answer = _generate_with_slm(answer_messages, max_new_tokens=MAX_NEW_TOKENS)

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

    log.info(f"startup complete — api is live  (top_k={TOP_K}, max_tokens={MAX_NEW_TOKENS}, hyde_weight={HYDE_WEIGHT}, min_sim={MIN_SIMILARITY})")
    yield
    log.info("shutting down")


app = FastAPI(title="Nepali Legal QA", version="2.0.0", lifespan=lifespan)

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
    log.info(f"done in {elapsed}s — answer length: {len(result['answer'])} chars")

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