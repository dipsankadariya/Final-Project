import logging
import os
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from auth import Token, TokenData, create_access_token, verify_access_token, verify_google_token


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


MODEL_ID = os.getenv("MODEL_ID", "zeri000/nepali_legal_qwen_merged_4")
DOC_FILE_PATH = os.getenv("DOC_FILE_PATH", "../../../augmented_nepali_legal_rag.txt")
GROQ_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"


app = FastAPI(title="Nepali Legal QA - HyDE vs Baseline RAG", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


tokenizer = None
model = None
vector_store = None
generators = []
terminators = None
request_counter = [0]


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    question: str
    hyde_passage: str
    baseline_retrieved_docs: list[str]
    hyde_retrieved_docs: list[str]
    baseline_answer: str
    hyde_answer: str
    processing_time: float


def build_terminators():
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = None

    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        ids.append(im_end_id)

    return list(dict.fromkeys(ids)) or None


def format_docs_for_prompt(docs: list) -> str:
    return "\n\n".join(
        f"[Context {idx}]\n{doc.page_content}"
        for idx, doc in enumerate(docs, start=1)
    )


def generate_hyde_document(user_query: str) -> str:
    """
    Fine-tuned SLM is used only for HyDE generation, matching the project design.
    """
    messages = [
        {
            "role": "system",
            "content": "तपाईं एक विशेषज्ञ नेपाली कानूनी सहायक हुनुहुन्छ।",
        },
        {
            "role": "user",
            "content": f"यस प्रश्नको आधारमा एउटा विस्तृत र सम्भावित कानूनी उत्तर वा व्याख्या तयार गर्नुहोस्: {user_query}",
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.95,
        do_sample=True,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def retrieve_with_baseline(question: str, top_k: int):
    return vector_store.similarity_search(question, k=top_k)


def retrieve_with_hyde(hyde_passage: str, top_k: int):
    return vector_store.similarity_search(hyde_passage, k=top_k)


def generate_answer_from_docs(question: str, docs: list, generator) -> str:
    response = generator.invoke(
        {
            "document": format_docs_for_prompt(docs),
            "query": question,
        }
    )
    return response.content.strip()


@app.on_event("startup")
def load_models():
    global tokenizer, model, vector_store, generators, terminators

    log.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    terminators = build_terminators()
    log.info("Loaded HyDE SLM: %s", MODEL_ID)

    if not os.path.exists(DOC_FILE_PATH):
        raise RuntimeError(f"Document file not found: {DOC_FILE_PATH}")

    loader = TextLoader(DOC_FILE_PATH, autodetect_encoding=True)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n"],
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    vector_store = FAISS.from_documents(chunks, embeddings)
    log.info("Built FAISS store with %d chunks", len(chunks))

    active_keys = [key for key in GROQ_KEYS if key]
    if not active_keys:
        raise RuntimeError("At least one GROQ_API_KEY environment variable is required.")

    system_prompt = """
You are a Nepali legal question-answering assistant.

Rules:
- Answer in Nepali only.
- Use the provided context as the basis of the answer.
- If the context is insufficient, say so clearly.
- Prefer a direct, complete answer.
- In your final answer please include the name of the law or the ain.
"""
    answer_prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("human", "Generate the best answer on the basis of this {document} for user query: {query}. Also include the best suitable ain and law that the document represent in the final answer."),
        ]
    )

    generators = [
        answer_prompt | ChatGroq(model="openai/gpt-oss-120b", api_key=key)
        for key in active_keys
    ]
    log.info("Initialized %d answer generator(s)", len(generators))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": device,
        "has_vector_store": vector_store is not None,
        "has_llm": bool(generators),
    }


class GoogleTokenRequest(BaseModel):
    token: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: TokenData


@app.post("/api/auth/google", response_model=AuthResponse)
def google_login(req: GoogleTokenRequest):
    """Login with Google OAuth2 token."""
    try:
        token_data = verify_google_token(req.token)
        access_token = create_access_token({
            "sub": token_data.sub,
            "email": token_data.email,
            "name": token_data.name,
            "picture": token_data.picture,
        })
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user=token_data
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/api/auth/verify")
def verify_token(authorization: Optional[str] = Header(None)):
    """Verify and get current user info from token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    try:
        # Extract token from "Bearer <token>"
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        
        token = parts[1]
        user = verify_access_token(token)
        return {"user": user}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


def get_current_user(authorization: Optional[str] = Header(None)) -> TokenData:
    """Dependency to verify token and get current user."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    try:
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        
        token = parts[1]
        return verify_access_token(token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest, authorization: Optional[str] = Header(None)):
    if model is None or vector_store is None or not generators:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    top_k = max(1, min(req.top_k or 3, 8))
    t0 = time.perf_counter()

    hyde_passage = generate_hyde_document(question)

    baseline_docs = retrieve_with_baseline(question, top_k)
    hyde_docs = retrieve_with_hyde(hyde_passage, top_k)

    if not baseline_docs or not hyde_docs:
        raise HTTPException(status_code=500, detail="Document retrieval failed")

    generator = generators[request_counter[0] % len(generators)]
    request_counter[0] += 1

    try:
        baseline_answer = generate_answer_from_docs(question, baseline_docs, generator)
    except Exception as exc:
        log.exception("Baseline answer generation failed")
        baseline_answer = f"Baseline answer error: {exc}"

    try:
        hyde_answer = generate_answer_from_docs(question, hyde_docs, generator)
    except Exception as exc:
        log.exception("HyDE answer generation failed")
        hyde_answer = f"HyDE answer error: {exc}"

    return QueryResponse(
        question=question,
        hyde_passage=hyde_passage,
        baseline_retrieved_docs=[doc.page_content for doc in baseline_docs],
        hyde_retrieved_docs=[doc.page_content for doc in hyde_docs],
        baseline_answer=baseline_answer,
        hyde_answer=hyde_answer,
        processing_time=round(time.perf_counter() - t0, 2),
    )
