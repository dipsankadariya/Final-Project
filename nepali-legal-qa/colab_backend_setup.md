# Colab backend setup for `rag-experiments`

## Cell 1 – Clone, checkout branch, install, ngrok

```python
# Fresh clone
!rm -rf Final-Project
!git clone https://github.com/dipsankadariya/Final-Project.git

# Repo root + experiment branch
%cd Final-Project
!git fetch origin
!git checkout rag-experiments
!git pull origin rag-experiments

# Backend folder
%cd nepali-legal-qa/backend

# Install deps
!pip install -r requirements.txt
!pip install pyngrok

# Clear old FAISS index
import os
for path in ["legal_faiss.index", "legal_docs.npy"]:
    if os.path.exists(path):
        os.remove(path)

# HF login
from huggingface_hub import login, HfFolder
login()  # paste token when asked

# Env for backend
hf_token = HfFolder.get_token()
os.environ["HF_TOKEN"] = hf_token
os.environ["MAX_NEW_TOKENS"] = "768"
os.environ["TOP_K"] = "5"
os.environ["MIN_SIM"] = "0.3"

# Ngrok
from pyngrok import ngrok
ngrok.set_auth_token("3BWKR1ZKqXMZGmiDFGef66UW34d_5MYW6PYcg888saxnn7Gbd")
public_url = ngrok.connect(8000, "http")
public_url
```

## Cell 2 – Start API

```python
!uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 120
```
