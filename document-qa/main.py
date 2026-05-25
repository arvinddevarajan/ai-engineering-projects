from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag import answer_question
import tempfile
import os

app = FastAPI(title="Document Q&A API")

class QuestionRequest(BaseModel):
    question: str
    filepath: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(request: QuestionRequest):
    result = answer_question(request.question, request.filepath)
    return result
