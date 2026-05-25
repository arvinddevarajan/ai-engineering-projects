import os
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import build_index, get_relevant_chunks
from models import run_all_models
from evaluator import evaluate

app = FastAPI(title="LLM Eval Harness")

ALLOWED_DIR = os.path.abspath("sample_docs")


class EvalRequest(BaseModel):
    filepath: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate_models(request: EvalRequest):
    filepath = os.path.abspath(request.filepath)
    if not filepath.startswith(ALLOWED_DIR):
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    try:
        chunks, embeddings = build_index(filepath)
        relevant_chunks = get_relevant_chunks(request.question, chunks, embeddings)
        context = "\n\n".join(relevant_chunks)

        results = run_all_models(request.question, context)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(evaluate, request.question, context, r["answer"])
                for r in results
            ]
            scores_list = [f.result() for f in futures]

        for result, scores in zip(results, scores_list):
            result["faithfulness"] = scores["faithfulness"]
            result["relevance"] = scores["relevance"]

        return {"question": request.question, "results": results}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))