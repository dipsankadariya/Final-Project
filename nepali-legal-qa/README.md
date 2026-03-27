# Nepali Legal QA — HyDE + RAG System

Intelligent Q&A system for Nepali law powered by a fine-tuned Qwen2.5 SLM, Hypothetical Document Embeddings (HyDE), and FAISS-backed Retrieval-Augmented Generation (RAG).

---

## Architecture

```
User Query
  │
  ▼
[SLM: Qwen2.5-1.5B fine-tuned]
    │  Generate hypothetical legal passage (HyDE)
    ▼
[multilingual-e5-base embedder]
    │  Embed hypothetical passage → query vector
    ▼
[FAISS Index]
    │  Top-K cosine similarity search
    ▼
[Retrieved Legal Passages]
  │  Context block
  ▼
[SLM: Qwen2.5-1.5B fine-tuned]
    │  Grounded final answer
    ▼
Final Legal Answer (Nepali)
```

### Models & Data
| Component | Model / Dataset |
|-----------|----------------|
| Fine-tuned SLM | `Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-merged` |
| Embedder | `intfloat/multilingual-e5-base` |
| Training dataset | `zeri000/augmented_nepali_legal_qa.csv` |
| Base model | `unsloth/Qwen2.5-3B-Instruct` (base used to create the 1.5B merged variant) |

---

## Setup

### 1. Backend

```bash
cd backend
python -m venv .venv311 && source .venv311/bin/activate  # Windows: .venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Environment variables** (optional — defaults shown):
```bash
export HF_TOKEN="hf_..."                          # Needed if model is gated
export MODEL_REPO="Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-merged"
export TOP_K=5
export MAX_NEW_TOKENS=512
```

**Run the server:**
```bash
python main.py
# Server starts at http://localhost:8000
```

> **GPU strongly recommended.** CPU inference is very slow (~5-10 min/query).
> Use the 1.5B model for lighter hardware; 3B for best quality.

First run will:
1. Download the SLM and embedder from HuggingFace (~3-7 GB)
2. Load the dataset and build a FAISS index (saved to `./legal_faiss.index`)

Subsequent runs load the index from disk (fast).

---

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

Create a frontend .env file from the example and adjust if needed:

```bash
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
```

The Vite dev server proxies `/api/*` to `http://localhost:8000` automatically.

For production build:
```bash
npm run build
# Output in frontend/dist/
```

---

## API Reference

### `POST /api/query`
```json
{
  "question": "नेपालको संविधान अनुसार नागरिकको मौलिक हक के हो?",
  "top_k": 5
}
```
**Response:**
```json
{
  "question": "...",
  "hyde_passage": "SLM-generated hypothetical answer used for retrieval",
  "retrieved_docs": ["doc1...", "doc2...", "..."],
  "answer": "Final grounded legal answer",
  "processing_time": 12.4
}
```

### `GET /api/health`
```json
{
  "status": "ok",
  "model": "Dipsan99/nepali-legal-hyde-qwen2.5-3b-merged",
  "index_size": 5432,
  "device": "cuda"
}
```

---

## Frontend Features

- **Swiss minimal design** — clean typography (Cormorant Garamond + Barlow), light palette
- **Pipeline visualization** — shows HyDE → Retrieval → Answer steps with progress
- **Collapsible panels** — HyDE passage and retrieved docs are expandable for transparency
- **Sample questions** — quick-launch chips in Nepali
- **Responsive** — works on mobile

---

## Project Structure

```
nepali-legal-qa/
├── backend/
│   ├── main.py            # FastAPI server — full HyDE-RAG pipeline
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Main React component + inline styles
│   │   └── main.jsx       # React entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js     # Vite + proxy config
└── README.md              # This file
```
