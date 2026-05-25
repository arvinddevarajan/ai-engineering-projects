import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_document(filepath):
    with open(filepath, "r") as f:
        return f.read()


def chunk_text(text, chunk_size=500, overlap=100):
    assert overlap < chunk_size, "overlap must be less than chunk_size"
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_relevant_chunks(question, chunks, chunk_embeddings, top_n=3):
    question_embedding = embed([question])[0]
    scored = [
        (cosine_similarity(question_embedding, ce), chunk)
        for chunk, ce in zip(chunks, chunk_embeddings)
    ]
    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:top_n]]


def build_index(filepath):
    text = load_document(filepath)
    chunks = chunk_text(text)
    embeddings = embed(chunks)
    return chunks, embeddings