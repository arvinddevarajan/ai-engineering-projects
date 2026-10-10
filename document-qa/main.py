from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_question
from agent import run_agent

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

@app.post("/agent/ask")
def agent_ask(request: QuestionRequest):
    return run_agent(request.question, request.filepath)
