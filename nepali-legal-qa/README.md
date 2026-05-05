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
Groq openai/gpt-oss-120b generates the final Nepali answer
(Round-robin across 4 API keys)
        │
        ▼
Return both baseline RAG and HyDE-enhanced results
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
│  └─ Cell 3: uvicorn main:app :8000  │       │  ├─ VITE_GOOGLE_CLIENT_ID   │
│                                     │       │  └─ /api/* proxied → Colab  │
│  backend/main.py  (FastAPI)         │       │                              │
│  ├─ GET  /api/health                │       │  frontend/src/               │
│  ├─ POST /api/auth/google           │       │  ├─ App.jsx (main UI)        │
│  ├─ GET  /api/auth/verify           │       │  ├─ GoogleLogin.jsx (OAuth)  │
│  └─ POST /api/query                 │       │  └─ index.css (Tailwind)    │
│      ├─ HyDE via local SLM          │       └──────────────────────────────┘
│      ├─ FAISS retrieval (LaBSE)     │
│      └─ Answer via Groq LLM         │
│      └─ JWT token verification      │
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
    │   ├── main.py               ← FastAPI app (HyDE pipeline + OAuth)
    │   ├── auth.py               ← JWT & Google OAuth verification
    │   ├── requirements.txt      ← Python dependencies
    │   └── .env.example          ← Config reference
    │
    └── frontend/
        ├── src/
        │   ├── App.jsx           ← React UI (query interface)
        │   ├── GoogleLogin.jsx    ← Google OAuth2 login component
        │   ├── main.jsx
        │   └── index.css          ← Tailwind CSS styles
        ├── .env                  ← Set VITE_API_BASE & VITE_GOOGLE_CLIENT_ID (gitignored)
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
| Answer LLM | `openai/gpt-oss-120b` via Groq (round-robin, 4 keys) |
| RAG corpus | `augmented_nepali_legal_rag.txt` (upload to Colab manually) |
| Eval dataset | `ragas_evaluation_simple_rag_165_datas.csv` (165 Q&A pairs) |
| Authentication | Google OAuth 2.0 + JWT tokens (24-hour expiry) |

**Fine-tuning setup:** LoRA (rank 16, alpha 32) — 1.78% of parameters trained. 3 epochs, ~10.5K samples, Tesla T4, ~2h 45min, final val loss ~0.415.

---

## ▶ How to run the application

### Prerequisites

| Tool | Where |
|---|---|
| Google account | For Colab GPU & OAuth2 authentication |
| Google OAuth 2.0 credentials | [Google Cloud Console](https://console.cloud.google.com) — for login |
| Groq API keys (×4) | [console.groq.com](https://console.groq.com) — free tier |
| ngrok account + token | [ngrok.com](https://ngrok.com) — free tier |
| Node.js 18+ | Local machine |
| `augmented_nepali_legal_rag.txt` | Must be uploaded to Colab manually |

---

### Step 0 — Set up Google OAuth 2.0 credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (or use an existing one)
3. Enable the **Google+ API**
4. Go to **APIs & Services → Credentials**
5. Create an **OAuth 2.0 Client ID** (Web application):
   - **Authorized redirect URIs:** `http://localhost:3000`
   - Copy the **Client ID** — you'll need it for the frontend `.env` file
6. Go to [Groq Console](https://console.groq.com) and create 4 API keys (free tier)
7. Set up an [ngrok account](https://ngrok.com) and get an auth token

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

5. **Run Cell 2** — set API keys, Google OAuth Client ID, and start ngrok:
   ```python
   import os
   from google.colab import userdata

   os.environ["GROQ_API_KEY"]   = userdata.get("GROQ_API_KEY")
   os.environ["GROQ_API_KEY_2"] = userdata.get("GROQ_API_KEY_2")
   os.environ["GROQ_API_KEY_3"] = userdata.get("GROQ_API_KEY_3")
   os.environ["GROQ_API_KEY_4"] = userdata.get("GROQ_API_KEY_4")
   os.environ["GOOGLE_CLIENT_ID"] = userdata.get("GOOGLE_CLIENT_ID")  # ← Add this
   os.environ["SECRET_KEY"] = "your-secret-key-change-in-production"   # ← Generate a random string

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

### Environment variables

**Backend** (set in Colab Cell 2):
- `GROQ_API_KEY`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3`, `GROQ_API_KEY_4` — Groq API keys
- `GOOGLE_CLIENT_ID` — Your Google OAuth 2.0 Client ID
- `SECRET_KEY` — JWT signing secret (generate a random string, e.g., `openssl rand -hex 32`)
- `MODEL_ID` — (optional) HuggingFace model ID (default: `zeri000/nepali_legal_qwen_merged_4`)
- `DOC_FILE_PATH` — (optional) Path to the RAG document file

**Frontend** (set in `frontend/.env`):
- `VITE_API_BASE` — Backend URL from ngrok (e.g., `https://xxxx-xxxx.ngrok-free.app`)
- `VITE_GOOGLE_CLIENT_ID` — Google OAuth 2.0 Client ID (must match backend)

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

3. Set the backend URL and Google OAuth credentials — edit `frontend/.env`:
   ```env
   VITE_API_BASE=https://xxxx-xxxx.ngrok-free.app
   VITE_GOOGLE_CLIENT_ID=your-google-client-id-here.apps.googleusercontent.com
   ```
   > ⚠️ **The ngrok URL changes every Colab session.** Update `.env` and restart `npm run dev` each time.
   > Get your Google Client ID from [Google Cloud Console](https://console.cloud.google.com) > APIs & Services > Credentials.

4. Start the dev server:
   ```bash
   npm run dev
   ```
   Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

### Authentication flow

```
Browser → Google Login Button (GoogleLogin.jsx)
              ↓
    User opens Google OAuth consent screen
              ↓
    Google returns ID token to frontend
              ↓
    Frontend sends ID token to /api/auth/google
              ↓
    Backend verifies token → creates JWT access token
              ↓
    Frontend stores JWT in localStorage
              ↓
    All subsequent /api/query calls include JWT in Authorization header
              ↓
    Backend middleware verifies token or allows anonymous access
```

Users can optionally log in with Google. The access token is valid for 24 hours.

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
    [3] Groq openai/gpt-oss-120b answers in Nepali
    [4] Return both baseline & HyDE results
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

### Authentication

#### `POST /api/auth/google`

Exchange a Google OAuth2 ID token for a JWT access token.

**Request:**
```json
{
  "token": "[google-id-token-from-google-login-sdk]"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer",
  "user": {
    "sub": "user-google-id",
    "email": "user@example.com",
    "name": "User Name",
    "picture": "https://..."
  }
}
```

#### `GET /api/auth/verify`

Verify the current JWT access token and get user info.

**Headers:**
```
Authorization: Bearer [access-token]
```

**Response:**
```json
{
  "user": {
    "sub": "user-google-id",
    "email": "user@example.com",
    "name": "User Name",
    "picture": "https://..."
  }
}
```

---

### `POST /api/query`

Query the Nepali legal QA system. Returns both baseline RAG and HyDE-enhanced RAG results for comparison.

**Headers:**
```
Authorization: Bearer [access-token]  (optional, but recommended)
Content-Type: application/json
```

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
  "hyde_passage": "...[hypothetical legal passage generated by fine-tuned SLM]...",
  "baseline_retrieved_docs": [
    "[retrieved chunk 1 using baseline retrieval]",
    "[retrieved chunk 2 using baseline retrieval]",
    "[retrieved chunk 3 using baseline retrieval]"
  ],
  "hyde_retrieved_docs": [
    "[retrieved chunk 1 using HyDE passage]",
    "[retrieved chunk 2 using HyDE passage]",
    "[retrieved chunk 3 using HyDE passage]"
  ],
  "baseline_answer": "...[final answer generated from baseline docs in Nepali]...",
  "hyde_answer": "...[final answer generated from HyDE docs in Nepali]...",
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