from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_document(filepath):
    with open(filepath, "r") as f:
        return f.read()

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def find_relevant_chunks(question, chunks, top_n=3):
    question_words = set(question.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:top_n]]

def answer_question(question, filepath):
    text = load_document(filepath)
    chunks = chunk_text(text)
    relevant = find_relevant_chunks(question, chunks)
    
    context = "\n\n".join(relevant)
    
    messages = [
        {"role": "system", "content": f"Answer based only on this context. If the answer is not in the context, say 'I don't have that information'.\n\nContext:\n{context}"},
        {"role": "user", "content": question}
    ]
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": relevant
    }
