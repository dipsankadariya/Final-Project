# 📚 Citations Quick Reference

## What Are Citations?

Citations show **exactly where** the answer comes from. Each citation has:

- **📄 Source**: The file name (e.g., `augmented_nepali_legal_rag.txt`)
- **#️⃣ Chunk #**: Which section of the document
- **📝 Content**: The actual legal text used

## How to Read Citations in the UI

### Baseline RAG Results
Shows documents retrieved using **direct keyword/semantic matching** on your question:
```
1️⃣ लोक सेवा आयोग...
   📄 augmented_nepali_legal_rag.txt
```
Click to expand and see:
- Full legal text
- Exact source
- Chunk number for reference

### HyDE RAG Results
Shows documents retrieved using **AI-generated hypothetical passages**:
```
1️⃣ पदाधिकारीको चयन...
   📄 augmented_nepali_legal_rag.txt
```
This approach often finds more relevant documents!

## Why Citations Matter

✅ **Transparency**: See the source material  
✅ **Verification**: Check if the answer is accurate  
✅ **Learning**: Understand how the system retrieved results  
✅ **Trust**: Know which law/regulation applies  

## Example Flow

1. **You ask**: "लोक सेवा आयोगको मुख्य काम के हो?"
2. **System generates HyDE**: "लोक सेवा आयोग नेपालमा पदाधिकारीको चयन गर्ने निकाय हो..."
3. **Retrieves matching passages**: Shows 3 citations with sources
4. **Generates answer**: Uses those citations as context
5. **You see**: The answer + the original citations it used

## API Details (For Developers)

Each citation object contains:
```json
{
  "content": "विस्तृत कानूनी पाठ...",
  "source": "augmented_nepali_legal_rag.txt",
  "chunk_index": 5
}
```

## Limitations to Know

- Citations come from `augmented_nepali_legal_rag.txt` only
- Chunk size is ~1000 tokens - citations may be truncated
- Source always shows the parent file name, not specific law names yet
- No direct links to external documents (improvement for future)

## Next Steps for Enhancement

Future versions could add:
- 📌 Specific law/article names in source field
- 🔗 Direct links to full legal documents
- ⭐ Citation confidence scores
- 🌐 Multi-document support
