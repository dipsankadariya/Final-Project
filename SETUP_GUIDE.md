# Nepali Legal QA System - Setup & Citations Guide

## ✅ What's Been Updated

### 1. **API Keys Configuration**
- Created `.env` file in `backend/` directory with all 4 GROQ API keys
- Backend now loads environment variables automatically using `python-dotenv`
- System will round-robin between keys for load balancing

### 2. **Document Source**
- System is configured to use `augmented_nepali_legal_rag.txt` 
- File path: `../../../augmented_nepali_legal_rag.txt` (relative to backend)
- Automatically chunked into 1000-token pieces with 200-token overlap

### 3. **Citations System** ✨
Citations are now fully integrated! Here's what you get:

#### Backend (Already Implemented)
- **Citation Model**: Each retrieved document includes:
  - `content`: The legal text excerpt
  - `source`: The filename (e.g., "augmented_nepali_legal_rag.txt")
  - `chunk_index`: Position of the chunk in the document

#### Frontend Updates
- **Enhanced DocsCard Component**: Now displays:
  - Document preview with truncation
  - Source file name inline (📄 indicator)
  - Expandable view showing:
    - Full source information
    - Chunk number
    - Complete legal text excerpt

## 🚀 How to Use

### Backend Setup
```bash
cd 1.5b/nepali-legal-qa/backend/

# Install dependencies (if not already done)
pip install -r requirements.txt

# Run the backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd 1.5b/nepali-legal-qa/frontend/

# Install dependencies
npm install

# Create .env file
echo "VITE_API_BASE=http://localhost:8000" > .env.local

# Run development server
npm run dev
```

## 📊 API Response Structure

The `/api/query` endpoint returns:

```json
{
  "question": "user's question",
  "hyde_passage": "generated legal passage for retrieval",
  "baseline_retrieved_docs": [
    {
      "content": "legal text excerpt...",
      "source": "augmented_nepali_legal_rag.txt",
      "chunk_index": 1
    },
    ...
  ],
  "hyde_retrieved_docs": [
    {
      "content": "legal text excerpt...",
      "source": "augmented_nepali_legal_rag.txt",
      "chunk_index": 3
    },
    ...
  ],
  "baseline_answer": "answer from baseline retrieval",
  "hyde_answer": "answer from HyDE retrieval",
  "processing_time": 12.45
}
```

## 🎯 How Citations Work

1. **Question Input**
   - User asks a question in Nepali/English

2. **HyDE Generation**
   - Fine-tuned model generates a hypothetical legal passage

3. **Dual Retrieval**
   - **Baseline**: Direct semantic search on user question
   - **HyDE**: Semantic search using generated passage
   - Each retrieval returns top-k documents with metadata

4. **Citation Display**
   - Frontend shows retrieved documents with source information
   - Users can expand to see full citation details
   - Source filename helps verify authenticity

5. **Answer Generation**
   - Answer is generated using the retrieved documents as context
   - Citations provide transparency on source material

## 🔧 Configuration

Environment variables in `.env`:
```
GROQ_API_KEY=...
GROQ_API_KEY_2=...
GROQ_API_KEY_3=...
GROQ_API_KEY_4=...
DOC_FILE_PATH=../../../augmented_nepali_legal_rag.txt
MODEL_ID=zeri000/nepali_legal_qwen_merged_4
```

## 💡 Key Features

- ✅ **Multi-key support**: 4 GROQ API keys for reliability
- ✅ **Full citations**: Source tracking for every retrieved passage
- ✅ **Chunk metadata**: Know exactly which chunk was used
- ✅ **Dual retrieval**: Compare baseline vs HyDE approaches
- ✅ **Low-resource**: Fine-tuned 1.5B model for efficient inference
- ✅ **Production-ready**: Error handling, CORS, authentication

## 📝 Notes

- All 4 API keys will be used in round-robin fashion
- Chunk size is 1000 tokens with 200-token overlap
- HyDE generation uses the fine-tuned model for better context
- Citations include both baseline and HyDE retrieved documents
