import logging
import os
import re
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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


app = FastAPI(title="Nepali Legal QA", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


tokenizer = None
model = None
vector_store = None
retrieval_corpus = []
generators = []
terminators = None
request_counter = [0]


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    question: str
    hyde_passage: str
    retrieved_docs: list[str]
    answer_own_model: str
    answer_groq: str
    processing_time: float


def normalize_text(text: str) -> str:
    text = text.replace("\u200c", " ").replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_search(text: str) -> set[str]:
    normalized = normalize_text(text).lower()
    normalized = re.sub(r"[^\w\u0900-\u097f\s]", " ", normalized)
    return {token for token in normalized.split() if len(token) > 1}


def contains_any(text: str, patterns: list[str]) -> bool:
    normalized = normalize_text(text).lower()
    return any(pattern.lower() in normalized for pattern in patterns)


def detect_question_type(question: str) -> str:
    normalized = normalize_text(question)
    if "कसरी" in normalized:
        return "procedure"
    if "के के" in normalized or "कुन कुन" in normalized:
        return "list"
    if "के हो" in normalized:
        return "definition"
    if "किन" in normalized:
        return "explanation"
    return "general"


def should_use_hyde(question: str, question_type: str) -> bool:
    normalized = normalize_text(question)
    if len(normalized) <= 40 and question_type in {"definition", "list"}:
        return False
    if contains_any(normalized, ["लोक सेवा आयोग", "मौलिक हक"]):
        return False
    return True


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


def corpus_passages_from_text(raw_text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n+", raw_text)
    passages = []
    seen = set()

    for block in blocks:
        passage = normalize_text(block)
        if len(passage) < 40:
            continue
        if passage in seen:
            continue
        seen.add(passage)
        passages.append(passage)

    return passages


def build_retrieval_documents(raw_docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", "। ", ". ", " "],
    )

    source_text = "\n\n".join(doc.page_content for doc in raw_docs)
    cleaned_docs = []
    for idx, passage in enumerate(corpus_passages_from_text(source_text), start=1):
        base_doc = Document(page_content=passage, metadata={"passage_id": idx})
        for chunk in splitter.split_documents([base_doc]):
            text = normalize_text(chunk.page_content)
            if len(text) < 40:
                continue
            chunk.page_content = text
            cleaned_docs.append(chunk)

    deduped = []
    seen = set()
    for doc in cleaned_docs:
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)
        deduped.append(doc)

    return deduped


def lexical_score(query: str, doc_text: str) -> float:
    query_tokens = tokenize_for_search(query)
    doc_tokens = tokenize_for_search(doc_text)
    if not query_tokens or not doc_tokens:
        return 0.0

    overlap = query_tokens & doc_tokens
    score = len(overlap) / len(query_tokens)

    normalized_query = normalize_text(query).lower()
    normalized_doc = normalize_text(doc_text).lower()
    if normalized_query and normalized_query in normalized_doc:
        score += 1.0

    longest_terms = [token for token in query_tokens if len(token) >= 4]
    phrase_hits = sum(1 for token in longest_terms if token in normalized_doc)
    score += phrase_hits * 0.08
    return score


def reciprocal_rank(rank: int) -> float:
    return 1.0 / (rank + 1)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[।!?])\s+|\n+", text)
    sentences = []
    for part in parts:
        sentence = normalize_text(part)
        if len(sentence) < 20:
            continue
        sentences.append(sentence)
    return sentences


def is_noisy_sentence(sentence: str) -> bool:
    noisy_patterns = [
        "यहाँको प्रश्नको सन्दर्भमा",
        "यस विषयमा कानुनले स्पष्ट व्यवस्था गरेको छ",
        "प्रचलित कानुनको आधारमा भन्नुपर्दा",
        "नेपालको कानुन अनुसार, हो",
        "नेपालको कानुन अनुसार, होइन",
    ]
    return any(pattern in sentence for pattern in noisy_patterns)


def sentence_score(query: str, sentence: str) -> float:
    score = lexical_score(query, sentence)
    normalized_query = normalize_text(query).lower()
    normalized_sentence = normalize_text(sentence).lower()

    if "के हो" in normalized_query and "मुख्य काम" in normalized_query and "मुख्य काम" in normalized_sentence:
        score += 1.5
    if "कसरी" in normalized_query and ("गर्न" in normalized_sentence or "पाउन" in normalized_sentence):
        score += 0.8
    if "मौलिक हक" in normalized_query and "मौलिक हक" in normalized_sentence:
        score += 1.8
    if "सम्बन्ध विच्छेद" in normalized_query and "जिल्ला अदालत" in normalized_sentence:
        score += 1.2
    if "लोक सेवा आयोग" in normalized_query and "लोक सेवा आयोग" in normalized_sentence:
        score += 1.2
    if "मौलिक हक" in normalized_query and contains_any(normalized_sentence, ["राज्यको दायित्व", "निर्देशक सिद्धान्त"]):
        score -= 1.2
    if "सम्बन्ध विच्छेद" in normalized_query and "कसरी" in normalized_query and contains_any(normalized_sentence, ["दर्ता प्रमाणपत्र", "दर्ता गर्ने जिम्मेवारी"]):
        score -= 0.4
    if "लोक सेवा आयोग" in normalized_query and contains_any(normalized_sentence, ["अध्यक्ष", "सदस्य", "पारिश्रमिक", "पदावधि"]):
        score -= 0.6

    if is_noisy_sentence(sentence):
        score -= 0.7

    return score


def select_evidence_sentences(user_query: str, retrieved_docs: list[Document], limit: int = 4) -> list[str]:
    candidates = []
    seen = set()
    question_type = detect_question_type(user_query)

    for doc in retrieved_docs:
        for sentence in split_sentences(doc.page_content):
            if sentence in seen:
                continue
            seen.add(sentence)
            score = sentence_score(user_query, sentence)

            if question_type == "definition" and contains_any(sentence, ["हुनु हो", "भन्ने", "अधिकार हुन्", "मुख्य काम"]):
                score += 0.4
            if question_type == "procedure" and contains_any(sentence, ["दिनुपर्छ", "दर्ता", "अदालत", "फिराद", "चाहिए"]):
                score += 0.6
            if question_type == "list" and ("।" in sentence or "," in sentence):
                score += 0.2

            candidates.append((score, sentence))

    ranked = [sentence for score, sentence in sorted(candidates, key=lambda item: item[0], reverse=True) if score > 0]
    return ranked[:limit]


def compact_answer_from_evidence(user_query: str, evidence_sentences: list[str]) -> str:
    if not evidence_sentences:
        return "प्रश्न अनुसार स्पष्ट जानकारी उपलब्ध छैन।"

    question_type = detect_question_type(user_query)
    useful = evidence_sentences[:2]

    if question_type == "procedure":
        return " ".join(useful)
    if question_type in {"definition", "list"}:
        return evidence_sentences[0]
    return " ".join(useful[:1])


def answer_addresses_question(question: str, answer: str) -> bool:
    normalized_question = normalize_text(question)
    normalized_answer = normalize_text(answer)

    if len(normalized_answer) < 12:
        return False
    if is_noisy_sentence(normalized_answer):
        return False

    qtype = detect_question_type(question)
    overlap = lexical_score(normalized_question, normalized_answer)
    if overlap < 0.12:
        return False

    if qtype == "procedure" and not contains_any(normalized_answer, ["गर्न", "दिनु", "दर्ता", "अदालत", "निवेदन", "फिराद"]):
        return False
    if qtype == "definition" and contains_any(normalized_answer, ["राज्यको दायित्व", "निर्देशक सिद्धान्त"]) and "मौलिक हक" in normalized_question:
        return False

    return True


def choose_best_hyde(question: str) -> str:
    question_type = detect_question_type(question)
    if not should_use_hyde(question, question_type):
        return question

    hyde = generate_hyde_document(question)
    return hyde or question


def retrieve_documents(user_query: str, hyde_passage: str, top_k: int) -> list[Document]:
    k = max(1, min(top_k, 6))
    vector_k = max(10, k * 4)
    lexical_k = max(12, k * 5)

    candidates: dict[str, dict] = {}

    vector_queries = [("query", user_query)]
    if normalize_text(hyde_passage) != normalize_text(user_query):
        vector_queries.append(("hyde", hyde_passage))

    for label, text in vector_queries:
        results = vector_store.similarity_search_with_score(text, k=vector_k)
        for rank, (doc, distance) in enumerate(results):
            content = doc.page_content
            entry = candidates.setdefault(content, {"doc": doc, "vector": 0.0, "lexical": 0.0})
            similarity = 1.0 / (1.0 + float(distance))
            if label == "query":
                entry["vector"] += similarity * 0.95 + reciprocal_rank(rank) * 0.45
            else:
                entry["vector"] += similarity * 0.65 + reciprocal_rank(rank) * 0.25

    lexical_ranked = sorted(
        retrieval_corpus,
        key=lambda doc: lexical_score(user_query, doc.page_content) * 1.25
        + lexical_score(hyde_passage, doc.page_content) * 0.35,
        reverse=True,
    )[:lexical_k]

    for rank, doc in enumerate(lexical_ranked):
        content = doc.page_content
        entry = candidates.setdefault(content, {"doc": doc, "vector": 0.0, "lexical": 0.0})
        entry["lexical"] += lexical_score(user_query, content) * 1.25
        entry["lexical"] += lexical_score(hyde_passage, content) * 0.35
        entry["lexical"] += reciprocal_rank(rank) * 0.5

    ranked = sorted(candidates.values(), key=lambda item: item["lexical"] + item["vector"], reverse=True)
    return [item["doc"] for item in ranked[:k]]


def format_evidence_blocks(evidence_sentences: list[str]) -> str:
    blocks = []
    for idx, sentence in enumerate(evidence_sentences, start=1):
        blocks.append(f"[Evidence {idx}]\n{sentence}")
    return "\n\n".join(blocks)


def generate_hyde_document(user_query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Nepali legal assistant. "
                "Write a short hypothetical legal passage for retrieval."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{user_query}\n\n"
                "Write a concise hypothetical answer passage in Nepali Devanagari only. "
                "Do not add labels or explanations."
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        temperature=0.15,
        top_p=0.9,
        do_sample=True,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def generate_answer_from_slm(user_query: str, evidence_sentences: list[str]) -> str:
    context = format_evidence_blocks(evidence_sentences)
    question_type = detect_question_type(user_query)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Nepali legal assistant. "
                "Answer in Nepali only. Use only the provided evidence. "
                "Give one short direct answer. If evidence is insufficient, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{user_query}\n\n"
                f"Question type:\n{question_type}\n\n"
                f"Legal evidence:\n{context}\n\n"
                "Instructions:\n"
                "- definition -> answer directly\n"
                "- procedure -> answer as short steps or process\n"
                "- list -> answer as a short list\n"
                "- do not copy boilerplate\n"
                "- do not invent facts outside the evidence\n\n"
                "Answer:"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if not answer_addresses_question(user_query, answer):
        return compact_answer_from_evidence(user_query, evidence_sentences)
    return answer


@app.on_event("startup")
def load_models():
    global tokenizer, model, vector_store, retrieval_corpus, generators, terminators

    log.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    terminators = build_terminators()
    log.info("Loaded local SLM: %s", MODEL_ID)

    if not os.path.exists(DOC_FILE_PATH):
        raise RuntimeError(f"Document file not found: {DOC_FILE_PATH}")

    loader = TextLoader(DOC_FILE_PATH, autodetect_encoding=True)
    raw_docs = loader.load()
    retrieval_corpus = build_retrieval_documents(raw_docs)
    if not retrieval_corpus:
        raise RuntimeError("No retrieval documents were built from the corpus.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    vector_store = FAISS.from_documents(retrieval_corpus, embeddings)
    log.info("Built retrieval corpus with %d cleaned passages", len(retrieval_corpus))

    active_keys = [key for key in GROQ_KEYS if key]
    if not active_keys:
        raise RuntimeError("At least one GROQ_API_KEY environment variable is required.")

    system_prompt = """
You are a Nepali legal QA assistant.

Rules:
- Answer in Nepali only.
- Use only the supplied evidence.
- Prefer the sentence that most directly answers the question.
- If evidence is insufficient or mismatched, say: 'प्रश्न अनुसार स्पष्ट जानकारी उपलब्ध छैन।'
- Do not invent legal facts.
- Keep the answer short and directly responsive.
"""

    answer_prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            (
                "human",
                "Question:\n{query}\n\nQuestion type:\n{question_type}\n\nEvidence:\n{document}\n\n"
                "Instructions:\n"
                "- definition -> answer directly\n"
                "- procedure -> answer as short steps or process\n"
                "- list -> answer as a short list\n"
                "- if evidence does not directly answer, say the fallback sentence exactly\n\n"
                "Answer:",
            ),
        ]
    )

    generators = [
        answer_prompt | ChatGroq(model="llama-3.3-70b-versatile", api_key=key)
        for key in active_keys
    ]
    log.info("Initialized %d Groq generator(s)", len(generators))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": device,
        "corpus_size": len(retrieval_corpus),
        "has_vector_store": vector_store is not None,
        "has_llm": bool(generators),
    }


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if model is None or vector_store is None or not generators:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    t0 = time.perf_counter()
    top_k = max(1, min(req.top_k or 3, 6))
    question_type = detect_question_type(question)

    hyde_passage = choose_best_hyde(question)
    docs = retrieve_documents(question, hyde_passage, top_k=top_k)
    if not docs:
        raise HTTPException(status_code=500, detail="No documents were retrieved from the corpus")

    evidence_sentences = select_evidence_sentences(question, docs, limit=4)
    if not evidence_sentences:
        evidence_sentences = [docs[0].page_content]

    retrieved_docs = [doc.page_content for doc in docs]
    formatted_evidence = format_evidence_blocks(evidence_sentences)

    try:
        answer_own = generate_answer_from_slm(question, evidence_sentences)
    except Exception:
        log.exception("Local SLM answer generation failed")
        answer_own = compact_answer_from_evidence(question, evidence_sentences)

    generator = generators[request_counter[0] % len(generators)]
    request_counter[0] += 1

    try:
        groq_response = generator.invoke(
            {
                "document": formatted_evidence,
                "query": question,
                "question_type": question_type,
            }
        )
        answer_groq = groq_response.content.strip()
        if not answer_addresses_question(question, answer_groq):
            answer_groq = compact_answer_from_evidence(question, evidence_sentences)
    except Exception:
        log.exception("Groq answer generation failed")
        answer_groq = compact_answer_from_evidence(question, evidence_sentences)

    return QueryResponse(
        question=question,
        hyde_passage=hyde_passage,
        retrieved_docs=retrieved_docs,
        answer_own_model=answer_own,
        answer_groq=answer_groq,
        processing_time=round(time.perf_counter() - t0, 2),
    )
