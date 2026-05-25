# LLM Eval Harness

A tool for evaluating and comparing LLM responses across multiple providers on RAG-based question answering tasks.

## What it does

- Loads a document and builds a semantic search index using OpenAI embeddings
- Retrieves the most relevant chunks for a given question using cosine similarity
- Runs the question through 4 models simultaneously (GPT-4o-mini, LLaMA 3.1 8b, LLaMA 3.3 70b, Qwen3-32b)
- Scores each answer on faithfulness and relevance using an LLM judge (G-Eval pattern)
- Reports quality scores, latency, and cost per model side-by-side

## Why I built this

Choosing the right LLM for a production use case is an engineering decision, not a gut feeling. This harness makes the cost/quality/latency trade-off measurable instead of assumed.

## Tech Stack

- Python, FastAPI
- OpenAI, Groq APIs
- RAG with semantic search (cosine similarity over OpenAI embeddings)
- LLM-as-judge evaluation (G-Eval pattern)
- Parallel model execution via ThreadPoolExecutor
- Rich for terminal output

## How to run

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

### CLI

```bash
python3 cli.py sample_docs/company_policy.txt "How many vacation days?"
```

### API

```bash
uvicorn main:app --reload
```

POST to `/evaluate`:

```json
{
  "filepath": "sample_docs/company_policy.txt",
  "question": "What is the expense limit?"
}
```

## Limitations

- **Judge bias**: GPT-4o-mini is used as the judge for all models including itself, which may inflate GPT scores. Production systems should use multiple judges or human-calibrated scoring.
- **In-memory index**: Embeddings are not persisted. For large documents or multi-user scenarios, a vector database (Pinecone, pgvector) would be needed.
- **No authentication**: The API accepts any filepath within `sample_docs/`. Production use would require auth and stricter input validation.