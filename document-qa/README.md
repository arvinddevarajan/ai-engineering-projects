# Document Q&A System

A RAG-based system that answers questions about any document using an LLM.

## What it does
- Loads a text document and splits it into chunks
- Finds the most relevant chunks for a given question
- Uses an LLM to answer based only on the document content
- Returns the answer with source excerpts for traceability

## Tech Stack
- Python
- FastAPI
- Groq API (LLaMA 3.1)
- RAG (Retrieval Augmented Generation)

## How to run

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
CLI

python3 cli.py sample_docs/company_policy.txt "How many vacation days?"
API

uvicorn main:app --reload
POST to /ask:


{
  "question": "What is the expense limit?",
  "filepath": "sample_docs/company_policy.txt"
}
Why I built this
Enterprise teams deal with large volumes of internal documents. This system
demonstrates how RAG can make any document instantly queryable without
fine-tuning a model.


