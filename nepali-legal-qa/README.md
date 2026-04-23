# Nepali Legal QA — Fine-tuned SLM + HyDE RAG

**Final Year College Project** — Research prototype exploring whether a domain fine-tuned SLM with HyDE retrieval gives better results than a standard RAG pipeline for Nepali legal question answering.

> **Note:** Any API keys visible in old notebook outputs were hardcoded for testing purposes only and have since been revoked.

---

## What is HyDE and why we used it

Standard RAG embeds the user's question directly and searches for similar passages. The problem is that a short Nepali question and a long legal answer sit in very different places in embedding space, so retrieval often misses relevant docs.

HyDE fixes this: instead of embedding the question, we first ask the SLM to generate a *hypothetical answer*, embed that, and use it to search. A hypothetical answer is semantically much closer to real answer passages — improving retrieval quality, especially in a low-resource language like Nepali.

---

## Pipeline

```
User Question (Nepali / English)
        │
        ▼
Fine-tuned SLM (zeri000/nepali_legal_qwen_merged_4)
generates a hypothetical legal passage             [HyDE]
        │
        ▼
sentence-transformers/LaBSE embeds the hypothetical passage
        │
        ▼
FAISS searches augmented_nepali_legal_rag.txt → Top 3 chunks
        │
        ▼
Groq llama-3.3-70b-versatile generates the final Nepali answer
(Round-robin across 4 API keys)
```

---

## Architecture

```
┌─────────────────────────────────────┐       ┌──────────────────────────────┐
│      GOOGLE COLAB  (GPU T4)         │       │      LOCAL MACHINE           │
│                                     │       │                              │
│  collab-backend.ipynb               │       │  frontend/  (React + Vite)   │
│  ├─ Cell 1: install deps            │       │  ├─ npm run dev → :3000      │
│  ├─ Cell 2: set env vars + ngrok    │◄──────┤  ├─ VITE_API_BASE = ngrok   │
│  └─ Cell 3: uvicorn main:app :8000  │       │  └─ /api/* proxied → Colab  │
│                                     │       └──────────────────────────────┘
│  backend/main.py  (FastAPI)         │
│  ├─ GET  /api/health                │
│  └─ POST /api/query                 │
│      ├─ HyDE via local SLM          │
│      ├─ FAISS retrieval (LaBSE)     │
│      └─ Answer via Groq LLM         │
└─────────────────────────────────────┘
```

---

## Project structure

```
College-project/
├── nepali_rag_qa.ipynb               ← Original RAG research notebook
├── Complete_slm_finetune.ipynb        ← SLM fine-tuning pipeline (Kaggle)
├── collab-backend.ipynb               ← Run this in Google Colab (GPU)
│
├── ragas_evaluation_simple_rag_165_datas.csv        ← RAGAS evaluation data
├── ragas_evaluation_qa_own_finetune_nepali_hyde_151.csv
├── ragas_evaluation_qa_nepali_hyde (1).csv
│
└── nepali-legal-qa/
    ├── README.md
    ├── .gitignore
    │
    ├── backend/
    │   ├── main.py               ← FastAPI app (exact pipeline from nepali_rag_qa.ipynb)
    │   ├── requirements.txt      ← Python dependencies
    │   └── .env.example          ← Config reference
    │
    └── frontend/
        ├── src/
        │   ├── App.jsx           ← React UI
        │   ├── main.jsx
        │   └── index.css
        ├── .env                  ← Set VITE_API_BASE here (gitignored)
        ├── .env.example          ← Frontend config reference
        ├── vite.config.js        ← Proxies /api/* to Colab backend
        └── package.json
```

---

## Models and data

| | |
|---|---|
| Base model | `unsloth/Qwen2.5-1.5B-Instruct` |
| Fine-tuned SLM | [zeri000/nepali_legal_qwen_merged_4](https://huggingface.co/zeri000/nepali_legal_qwen_merged_4) |
| SLM role | HyDE hypothetical-passage generation only |
| Embedding model | `sentence-transformers/LaBSE` |
| Answer LLM | `llama-3.3-70b-versatile` via Groq (round-robin, 4 keys) |
| RAG corpus | `augmented_nepali_legal_rag.txt` (upload to Colab manually) |
| Eval dataset | `ragas_evaluation_simple_rag_165_datas.csv` (165 Q&A pairs) |

**Fine-tuning setup:** LoRA (rank 16, alpha 32) — 1.78% of parameters trained. 3 epochs, ~10.5K samples, Tesla T4, ~2h 45min, final val loss ~0.415.

---

## ▶ How to run the application

### Prerequisites

| Tool | Where |
|---|---|
| Google account | For Colab GPU |
| Groq API keys (×4) | [console.groq.com](https://console.groq.com) — free tier |
| ngrok account + token | [ngrok.com](https://ngrok.com) — free tier |
| Node.js 18+ | Local machine |
| `augmented_nepali_legal_rag.txt` | Must be uploaded to Colab manually |

---

### Step 1 — Start the backend in Google Colab

1. Open **[collab-backend.ipynb](./collab-backend.ipynb)** in [Google Colab](https://colab.research.google.com)

2. Set runtime: **Runtime → Change runtime type → GPU (T4)**

3. **Run Cell 1** — clones the repo and installs dependencies:
   ```python
   !git clone https://github.com/dipsankadariya/College-project.git
   %cd College-project/backend
   !pip install -r requirements.txt
   !pip install pyngrok
   ```
   When the HuggingFace login prompt appears, paste a **read-access HF token**.

4. **Upload `augmented_nepali_legal_rag.txt`** to `/content/` in Colab
   (Files panel on the left → Upload)

5. **Run Cell 2** — set API keys and start ngrok:
   ```python
   import os
   from google.colab import userdata

   os.environ["GROQ_API_KEY"]   = userdata.get("GROQ_API_KEY")
   os.environ["GROQ_API_KEY_2"] = userdata.get("GROQ_API_KEY_2")
   os.environ["GROQ_API_KEY_3"] = userdata.get("GROQ_API_KEY_3")
   os.environ["GROQ_API_KEY_4"] = userdata.get("GROQ_API_KEY_4")

   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")   # ← paste your token
   public_url = ngrok.connect(8000, "http")
   print(public_url)   # e.g. https://xxxx-xxxx.ngrok-free.app
   ```
   📋 **Copy this URL** — you need it in Step 2.

6. **Run Cell 3** — start the FastAPI server (keep this cell running):
   ```
   !uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 120
   ```
   Wait for: `Application startup complete.`

7. **Verify** the backend is live:
   ```bash
   curl https://xxxx-xxxx.ngrok-free.app/api/health
   ```
   Expected response:
   ```json
   {"status":"ok","model":"zeri000/nepali_legal_qwen_merged_4","device":"cuda","has_vector_store":true,"has_llm":true}
   ```

---

### Step 2 — Start the frontend locally

1. Open a terminal, navigate to the frontend directory:
   ```bash
   cd nepali-legal-qa/frontend
   ```

2. Install dependencies (first time only):
   ```bash
   npm install
   ```

3. Set the backend URL — edit `frontend/.env`:
   ```env
   VITE_API_BASE=https://xxxx-xxxx.ngrok-free.app
   ```
   > ⚠️ **This URL changes every Colab session.** Update `.env` and restart `npm run dev` each time.

4. Start the dev server:
   ```bash
   npm run dev
   ```
   Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

### How requests flow

```
Browser → localhost:3000/api/query
              ↓
    Vite dev server (proxy)
              ↓
    https://xxxx.ngrok-free.app/api/query
              ↓
    FastAPI on Google Colab (:8000)
              ↓
    [1] Local SLM generates HyDE passage (CUDA)
    [2] LaBSE embeds it → FAISS retrieves top 3 chunks
    [3] Groq llama-3.3-70b answers in Nepali
```

The Vite proxy handles CORS and ngrok headers automatically.

---

## API reference

### `GET /api/health`
```json
{
  "status": "ok",
  "model": "zeri000/nepali_legal_qwen_merged_4",
  "device": "cuda",
  "has_vector_store": true,
  "has_llm": true
}
```

### `POST /api/query`

**Request:**
```json
{
  "question": "नेपालमा सम्बन्ध विच्छेद कसरी गर्ने?",
  "top_k": 3
}
```

**Response:**
```json
{
  "question": "नेपालमा सम्बन्ध विच्छेद कसरी गर्ने?",
  "hyde_passage": "...[hypothetical legal passage from SLM]...",
  "retrieved_docs": ["[chunk 1]", "[chunk 2]", "[chunk 3]"],
  "answer": "...[final answer in Nepali from Groq]...",
  "processing_time": 14.2
}
```

---

## Known limitations

- SLM occasionally loops or hallucinates on complex legal queries — known with small models on low-resource languages
- FAISS index is built from QA-format data, so retrieval quality is bounded by corpus coverage
- ngrok free-tier URL expires on every Colab session restart

---

## Contributors

| | |
|---|---|
| **Dipsan Kadariya** | SLM fine-tuning · Frontend (`Complete_slm_finetune.ipynb`, `frontend/`) |
| **Ritesh Raut** | HyDE-RAG pipeline · Backend (`nepali_rag_qa.ipynb`, `backend/`) |