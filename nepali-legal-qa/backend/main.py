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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

os.environ["MAX_NEW_TOKENS"] = "1024"
os.environ["TOP_K"] = "5"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_REPO = os.environ.get("MODEL_REPO", "Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-merged")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
DATASET_NAME = os.environ.get("DATASET_NAME", "zeri000/augmented_nepali_legal_qa.csv")
INDEX_PATH = os.environ.get("INDEX_PATH", "./legal_faiss.index")
DOCS_PATH = os.environ.get("DOCS_PATH", "./legal_docs.npy")
TOP_K = int(os.environ.get("TOP_K", "3"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

SYSTEM_PROMPT = (
    "तपाईं नेपलको एक वरिष्ठ कानूनी विशेषज्ञ हुनुहुन्छ। "
    "प्रश्नको उत्तर दिँदा: "
    "१) सम्बन्धित ऐन वा धारा उल्लेख गर्नुहोस्, "
    "२) स्पष्ट र संक्षिप्त भाषामा व्याख्या गर्नुहोस्, " 
    "३) व्यावहारिक उदाहरण दिनुहोस् यदि सम्भव भए। "
    "उत्तर औपचारिक नेपाली कानूनी भाषामा दिनुहोस्।"
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")

    log.info(f"loading embedder: {EMBED_MODEL}")
    state["embedder"] = SentenceTransformer(EMBED_MODEL, device=device)
    embed_dim = state["embedder"].get_sentence_embedding_dimension()
    log.info(f"embedder ready dim={embed_dim}")

    log.info(f"loading slm: {MODEL_REPO}")
    tok_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_REPO, **tok_kwargs)
    if device == "cuda":
        state["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.float16,
            device_map="auto",
            **tok_kwargs,
        )
    else:
        state["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            **tok_kwargs,
        )
    state["model"].eval()
    log.info("slm ready")

    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        log.info("loading existing faiss index from disk")
        state["index"] = faiss.read_index(INDEX_PATH)
        state["docs"] = np.load(DOCS_PATH, allow_pickle=True).tolist()
        log.info(f"index loaded {state['index'].ntotal} vectors")
    else:
        log.info("building faiss index from dataset")
        from datasets import load_dataset
        ds = load_dataset(DATASET_NAME, split="train")
        docs = []
        for row in ds:
            instruction = str(row.get("instruction", "")).strip()
            output      = str(row.get("output", "")).strip()
            context     = str(row.get("input", "")).strip()
            passage     = instruction
            if context and context.lower() != "nan":
                passage += f"\n{context}"
            passage += f"\n{output}"
            docs.append(passage)

        log.info(f"embedding {len(docs)} passages")
        BATCH = 64
        all_embs = []
        for i in range(0, len(docs), BATCH):
            batch   = docs[i : i + BATCH]
            prefixed = [f"passage: {d}" for d in batch]
            embs = state["embedder"].encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
            all_embs.append(embs)
            if i % (BATCH * 10) == 0:
                log.info(f"embedded {min(i + BATCH, len(docs))} / {len(docs)}")

        embeddings = np.vstack(all_embs).astype("float32")
        state["index"] = faiss.IndexFlatIP(embed_dim)
        state["index"].add(embeddings)
        state["docs"] = docs

        faiss.write_index(state["index"], INDEX_PATH)
        np.save(DOCS_PATH, np.array(docs, dtype=object))
        log.info(f"faiss index built and saved {state['index'].ntotal} vectors")

    log.info("all models ready api is live")
    yield
    log.info("shutting down")


app = FastAPI(title="Nepali Legal QA API", version="1.0.0", lifespan=lifespan)

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
    question: str
    hyde_passage: str
    retrieved_docs: List[str]
    answer: str
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    model: str
    index_size: int
    device: str


def _terminators():
    tok = state["tokenizer"]
    ids = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end and im_end != tok.unk_token_id:
        ids.append(im_end)
    return ids


def generate_text(messages: list, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    tok   = state["tokenizer"]
    mdl   = state["model"]
    prompt    = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs    = tok(prompt, return_tensors="pt").to(mdl.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.01,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.2,
            eos_token_id=_terminators(),
            use_cache=True,
        )

    new_tokens = output_ids[0][input_len:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def hyde_retrieve(question: str, top_k: int) -> tuple[str, list[str]]:
    hyde_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    hypothetical = generate_text(hyde_messages, max_new_tokens=256)

    query_emb = state["embedder"].encode(
        [f"query: {hypothetical}"],
        normalize_embeddings=True,
    ).astype("float32")

    _, indices = state["index"].search(query_emb, top_k)
    retrieved  = [state["docs"][i] for i in indices[0] if i < len(state["docs"])]
    return hypothetical, retrieved


@app.get("/api/health", response_model=HealthResponse)
def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="ok",
        model=MODEL_REPO,
        index_size=state.get("index", faiss.IndexFlatIP(1)).ntotal,
        device=device,
    )


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    t0 = time.time()
    log.info(f"query: {req.question[:80]}")

    hyde_passage, retrieved_docs = hyde_retrieve(req.question, top_k=req.top_k)
    context_block = "\n\n---\n\n".join(
        [f"[सन्दर्भ {i+1}]\n{doc}" for i, doc in enumerate(retrieved_docs)]
    )
    answer_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"{req.question.strip()}"
            f"\n\nतलका कानूनी सन्दर्भहरूको आधारमा विस्तृत उत्तर दिनुहोस्:\n\n{context_block}"
        )},
    ]
    answer = generate_text(answer_messages, max_new_tokens=MAX_NEW_TOKENS)
    elapsed = round(time.time() - t0, 2)
    log.info(f"answer generated in {elapsed}s")

    return QueryResponse(
        question=req.question,
        hyde_passage=hyde_passage,
        retrieved_docs=retrieved_docs,
        answer=answer,
        processing_time=elapsed,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
