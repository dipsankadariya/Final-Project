# ================================================================
# BACKEND — FastAPI server for Nepali Legal QA
# Mirrors the exact pipeline from nepali_rag_qa.ipynb:
#   - HyDE generation  : fine-tuned SLM (zeri000/nepali_legal_qwen_merged_4)
#   - Embeddings       : sentence-transformers/LaBSE
#   - Vector store     : LangChain FAISS from augmented_nepali_legal_rag.txt
#                        (chunk_size=1000, chunk_overlap=200)
#   - Answer generation: Groq llama-3.3-70b-versatile (round-robin, 4 keys)
#
# Run in Google Colab with:
#   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
# ================================================================

import os
import time
import logging
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# LangChain / LangGraph — exact imports from nepali_rag_qa.ipynb
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = os.getenv("MODEL_ID", "zeri000/nepali_legal_qwen_merged_4")
DOC_FILE_PATH = os.getenv("DOC_FILE_PATH", "../../../augmented_nepali_legal_rag.txt")

# Groq API keys — env vars take priority, hardcoded values are the fallback
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "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")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", "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")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3", "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")
GROQ_API_KEY_4 = os.getenv("GROQ_API_KEY_4", "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")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Nepali Legal QA — HyDE RAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (populated on startup) ──────────────────────────────────────
tokenizer    = None
model        = None
retriever    = None
vector_store = None
generator    = None   # Groq llm_1  (answer prompt chain)
generator_2  = None   # Groq llm_2
generator_3  = None   # Groq llm_3
device       = "cuda" if torch.cuda.is_available() else "cpu"
terminators  = None

# Round-robin counter — same logic as the notebook's /ask endpoint
request_counter = [0]


def _build_terminators():
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        ids.append(im_end_id)

    return list(dict.fromkeys(ids)) or None


def _format_context_blocks(retrieved_docs: list) -> str:
    return "\n\n".join(
        f"[सन्दर्भ {i}]\n{doc.page_content.strip()}"
        for i, doc in enumerate(retrieved_docs, start=1)
    )


def retrieve_documents(user_query: str, hyde_passage: str, top_k: int):
    """
    Merge retrieval from HyDE and the original user query.
    This helps when the HyDE passage drifts semantically.
    """
    k = max(1, min(top_k, 8))
    candidates = []
    candidates.extend(vector_store.similarity_search(hyde_passage, k=k))
    candidates.extend(vector_store.similarity_search(user_query, k=k))

    deduped = []
    seen = set()
    for doc in candidates:
        content = doc.page_content.strip()
        if not content or content in seen:
            continue
        seen.add(content)
        deduped.append(doc)
        if len(deduped) >= k:
            break

    return deduped


# ── HyDE: generate hypothetical document with local SLM ──────────────────────
def generate_hyde_document(user_query: str) -> str:
    """
    Exact copy of generate_hyde_document() from nepali_rag_qa.ipynb.
    Uses the fine-tuned Qwen2.5 SLM to produce a hypothetical legal passage
    that is then embedded and used to retrieve real documents from FAISS.
    """
    messages = [
        {
            "role": "system",
            "content": "तपाईं एक विशेषज्ञ नेपाली कानूनी सहायक हुनुहुन्छ।",
        },
        {
            "role": "user",
            "content": (
                "यस प्रश्नको आधारमा एउटा विस्तृत र सम्भावित कानूनी उत्तर वा "
                f"व्याख्या तयार गर्नुहोस्: {user_query}"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
    input_length     = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ── Local SLM: generate answer using fine-tuned model ────────────────────────────
def generate_answer_from_slm(user_query: str, retrieved_docs: list) -> str:
    """
    Uses the fine-tuned SLM to generate a legal answer based on retrieved documents.
    Uses the exact ChatML format the model was fine-tuned on (matches Ritesh_fine_tune notebook).
    """
    # Format documents for context (same as fine-tuning format)
    context = _format_context_blocks(retrieved_docs)
    
    # Build user message exactly as fine-tuning format: question + optional context
    user_text = user_query
    if context:
        user_text += f"\n\nसन्दर्भ (Context):\n{context}"
    
    # Use ChatML format with proper roles (exact fine-tuning setup)
    messages = [
        {
            "role": "system",
            "content": "तपाईं एक विशेषज्ञ नेपाली कानूनी सहायक हुनुहुन्छ।",
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]
    
    # Apply chat template with add_generation_prompt=True (matches fine-tuning inference)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    # Generate with parameters from Ritesh_fine_tune inference test
    outputs = model.generate(
        **inputs,
        max_new_tokens=384,
        temperature=0.01,          # Ultra-low: factual legal information recall
        top_p=0.95,
        repetition_penalty=1.0,    # Prevent repetition in Nepali text
        do_sample=True,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_length     = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ── Startup: load models and build pipeline ───────────────────────────────────
@app.on_event("startup")
def load_models():
    global tokenizer, model, retriever, vector_store, terminators
    global generator, generator_2, generator_3

    log.info("Device: %s", device)

    # 1. Fine-tuned SLM — for HyDE generation only
    log.info("Loading tokenizer: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    log.info("Loading model: %s", MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",   # uses GPU automatically in Colab
    )
    terminators = _build_terminators()
    log.info("SLM loaded ✓")

    # 2. Build FAISS vector store — exact pipeline from nepali_rag_qa.ipynb
    log.info("Loading embeddings: sentence-transformers/LaBSE")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

    # Load corpus from local .txt file only
    local_txt = DOC_FILE_PATH
    if not os.path.exists(local_txt):
        raise RuntimeError(
            f"Document file not found at: {local_txt}\n"
            f"Please ensure augmented_nepali_legal_rag.txt exists at the correct location."
        )
    
    log.info("Loading documents from local file: %s", local_txt)
    loader  = TextLoader(local_txt)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n"]
    )
    chunks = splitter.split_documents(raw_docs)
    log.info("Loaded %d chunks from file", len(chunks))

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever    = vector_store.as_retriever(search_kwargs={"k": 3})
    log.info("FAISS index ready ✓")

    # 3. Groq LLMs — llama-3.3-70b-versatile, 4 keys for rate-limit headroom
    groq_model = "llama-3.3-70b-versatile"
    llm   = ChatGroq(model=groq_model, api_key=GROQ_API_KEY)
    llm_2 = ChatGroq(model=groq_model, api_key=GROQ_API_KEY_2)
    llm_3 = ChatGroq(model=groq_model, api_key=GROQ_API_KEY_3)
    llm_4 = ChatGroq(model=groq_model, api_key=GROQ_API_KEY_4)  # noqa: F841 (kept for parity)

    # 4. Answer-generation prompt chains — exact from nepali_rag_qa.ipynb
    system = """
तपाईं नेपाली कानुनी प्रश्नहरूको उत्तर दिने सहायक हुनुहुन्छ।

नियमहरू:
- उत्तर नेपालीमा मात्र दिनुहोस्।
- दिइएको सन्दर्भभित्रको जानकारीलाई मात्र आधार बनाउनुहोस्।
- सन्दर्भले स्पष्ट उत्तर नदिएमा 'दिइएको सन्दर्भका आधारमा स्पष्ट उत्तर भेटिएन' भन्नुहोस्।
- अनुमान, अतिरिक्त कानुनी तथ्य, वा बनावटी विवरण नथप्नुहोस्।
- सम्भव भएसम्म छोटो, स्पष्ट, र कानुनी रूपमा उपयोगी उत्तर दिनुहोस्।
- उत्तरमा सम्बन्धित सन्दर्भ नम्बर जस्तै [सन्दर्भ 1] उल्लेख गर्नुहोस्।
"""
    answer_prompt = ChatPromptTemplate(
        [
            ("system", system),
            (
                "human",
                "प्रश्न:\n{query}\n\nकानुनी सन्दर्भ:\n{document}\n\n"
                "माथिको सन्दर्भका आधारमा मात्र उत्तर दिनुहोस्।",
            ),
        ]
    )
    generator   = answer_prompt | llm
    generator_2 = answer_prompt | llm_2
    generator_3 = answer_prompt | llm_3

    log.info("RAG pipeline ready ✓")


# ── API schemas ───────────────────────────────────────────────────────────────
class QueryRequest(PydanticBaseModel):
    question: str
    top_k: Optional[int] = 3   # default k=3, matches notebook


class QueryResponse(PydanticBaseModel):
    question: str
    hyde_passage: str
    retrieved_docs: list[str]
    answer_own_model: str      # Answer from fine-tuned SLM
    answer_groq: str           # Answer from Groq LLM
    processing_time: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": device,
        "has_vector_store": retriever is not None,
        "has_llm": generator is not None,
    }


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if generator is None or retriever is None or model is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    t0 = time.perf_counter()

    # Step 1 — HyDE: local SLM generates a hypothetical legal passage
    hyde_passage = generate_hyde_document(req.question)

    # Step 2 — Retrieval: embed HyDE passage, search FAISS
    top_k = max(1, min(req.top_k or 3, 8))
    docs = retrieve_documents(req.question, hyde_passage, top_k=top_k)
    context = [doc.page_content for doc in docs]
    formatted_context = _format_context_blocks(docs)

    # Step 3a — Answer from own model (fine-tuned SLM)
    try:
        answer_own = generate_answer_from_slm(req.question, docs)
        log.info("Own model answer generated ✓")
    except Exception as e:
        log.error(f"Error generating answer from own model: {e}")
        answer_own = f"Error: {str(e)}"

    # Step 3b — Answer from Groq (round-robin across API keys)
    i = request_counter[0]
    request_counter[0] += 1

    groq_generators = [generator, generator_2, generator_3]
    active_generator = groq_generators[i % len(groq_generators)]
    response = active_generator.invoke({"document": formatted_context, "query": req.question})

    answer_groq = response.content

    processing_time = round(time.perf_counter() - t0, 2)
    return QueryResponse(
        question=req.question,
        hyde_passage=hyde_passage,
        retrieved_docs=context,
        answer_own_model=answer_own,
        answer_groq=answer_groq,
        processing_time=processing_time,
    )
