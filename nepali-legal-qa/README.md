# Nepali Legal QA — Fine-tuned SLM + HyDE RAG

**Final Year College Project** | Research prototype exploring whether a domain fine-tuned SLM with HyDE retrieval gives better results than a standard RAG pipeline for Nepali legal question answering.

> **Note:** Any HuggingFace API keys visible in the notebooks were hardcoded for testing purposes only and have since been revoked. Do not use them.

> This is an active research project. Results are preliminary and we're continuously working to improve them...

---

### What is HyDE and why we used it

Standard RAG embeds the user's question and searches for similar passages. The problem is that a short Nepali question and a long legal answer sit in very different places in embedding space, so retrieval often misses relevant docs.

HyDE fixes this: instead of embedding the question, we first ask the SLM to generate a *hypothetical answer*, embed that, and use it to search. A hypothetical answer is semantically much closer to real answer passages — improving retrieval quality, especially in a low-resource language like Nepali.

---

### How the full system works

```
User Question (Nepali)
    ↓
Fine-tuned SLM generates a hypothetical answer passage   [HyDE]
    ↓
multilingual-e5-base embeds the hypothetical passage
    ↓
FAISS searches 11K legal passages → Top 5 retrieved
    ↓
Fine-tuned SLM generates final answer using retrieved context
```

---

### Project structure

```
├── Complete_slm_finetune.ipynb   # Fine-tuning pipeline (Kaggle)
├── nepali_legal_rag.ipynb        # HyDE-RAG inference (Colab)
├── backend/
│   └── main.py                   # FastAPI server
└── frontend/
    └── src/App.jsx               # React + Vite + Tailwind UI
```

---

### Models and data

| | |
|---|---|
| Base model | `unsloth/Qwen2.5-1.5B-Instruct` |
| Fine-tuned model | [Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-merged](https://huggingface.co/Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-merged) |
| LoRA adapter | [Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-adapter](https://huggingface.co/Dipsan99/nepali-legal-hyde-qwen2.5-1.5b-adapter) |
| Embedding model | `intfloat/multilingual-e5-base` |
| Dataset | [zeri000/augmented_nepali_legal_qa.csv](https://huggingface.co/datasets/zeri000/augmented_nepali_legal_qa.csv) — 11,080 Nepali legal QA pairs |

**Fine-tuning setup:** LoRA (rank 16, alpha 32) on all attention + MLP layers — only 1.78% of parameters trained. 3 epochs on ~10.5K samples, Tesla T4, ~2h 45min. Final val loss ~0.415.

---

### Known limitations

- Model occasionally loops or hallucinates on complex legal queries — a known issue with smaller models on domain-specific low-resource languages
- FAISS index is built from a QA dataset, not raw legal documents, so retrieval quality is bounded by dataset coverage
- Limited by free-tier GPU compute throughout development

---

### Contributors

| | |
|---|---|
| **Dipsan Kadariya** | SLM fine-tuning and frontend  (`Complete_slm_finetune.ipynb`, `frontend/`) |
| **Ritesh Raut** | HyDE-RAG pipeline and backend(`nepali_legal_rag.ipynb`, `backend/`) |