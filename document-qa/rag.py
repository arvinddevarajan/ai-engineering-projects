from groq import Groq
from dotenv import load_dotenv
from openai import OpenAI
import os
import hashlib
import chromadb

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="document_qa",
    metadata={"hnsw:space": "cosine"}
)

def load_document(filepath):
    with open(filepath, "r") as f:
        return f.read()
# Previous version: fixed-size word chunking
# Kept temporarily for learning/reference

# def chunk_text(text, chunk_size=500):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size):
#         chunk = " ".join(words[i:i + chunk_size])
#         chunks.append(chunk)
#     return chunks

def chunk_text(text):
    chunks = [section.strip() for section in text.split("\n\n") if section.strip()]
    return chunks
# Previous version: keyword-overlap retrieval
# Replaced by embedding retrieval with Chroma

# def find_relevant_chunks(question, chunks, top_n=3):
#     question_words = set(question.lower().split())
#     scored = []
#     for chunk in chunks:
#         chunk_words = set(chunk.lower().split())
#         score = len(question_words & chunk_words)
#         scored.append((score, chunk))
#     scored.sort(reverse=True)
#     return [chunk for _, chunk in scored[:top_n]]

# def find_relevant_chunks(question, chunks, top_n=3):
#     if not chunks:
#         return []

#     chunk_response = embedding_client.embeddings.create(
#         model="text-embedding-3-small",
#         input=chunks
#     )
#     question_response = embedding_client.embeddings.create(
#         model="text-embedding-3-small",
#         input=question
#     )

#     question_vector = question_response.data[0].embedding
#     scored = []

#     for chunk, result in zip(chunks, chunk_response.data):
#         chunk_vector = result.embedding

#         dot_product = sum(
#             q * c for q, c in zip(question_vector, chunk_vector)
#         )
#         q_length = math.sqrt(sum(q * q for q in question_vector))
#         c_length = math.sqrt(sum(c * c for c in chunk_vector))
#         similarity = dot_product / (q_length * c_length)

#         scored.append((similarity, chunk))

#     scored.sort(key=lambda item: item[0], reverse=True)
#     return [chunk for _, chunk in scored[:top_n]]
def find_relevant_chunks(question, chunks, filepath, top_n=1):
    if not chunks:
        return []

    source = os.path.abspath(filepath)
    content_hash = hashlib.sha256(
        "\n\n".join(chunks).encode("utf-8")
    ).hexdigest()
    source_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    existing = collection.get(
        where={"source": source},
        include=["metadatas"]
    )

    already_indexed = (
        len(existing["ids"]) == len(chunks)
        and all(
            metadata.get("content_hash") == content_hash
            for metadata in existing["metadatas"]
        )
    )

    if not already_indexed:
        if existing["ids"]:
            collection.delete(where={"source": source})

        chunk_response = embedding_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunks
        )

        collection.upsert(
            ids=[f"{source_id}_{content_hash}_{i}" for i in range(len(chunks))],
            embeddings=[item.embedding for item in chunk_response.data],
            documents=chunks,
            metadatas=[
                {
                    "source": source,
                    "content_hash": content_hash,
                    "chunk_index": i
                }
                for i in range(len(chunks))
            ]
        )

    question_response = embedding_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    question_vector = question_response.data[0].embedding

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(top_n, len(chunks)),
        where={"source": source},
        include=["documents", "distances"]
    )

    print("Retrieved distances:", results["distances"][0])
    return results["documents"][0]

def answer_question(question, filepath):
    text = load_document(filepath)
    chunks = chunk_text(text)
    relevant = find_relevant_chunks(question, chunks, filepath)

    context = "\n\n".join(relevant)

    messages = [
        {"role": "system", "content": f"Answer based only on this context. If the answer is not in the context, say 'I don't have that information'.\n\nContext:\n{context}"},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": relevant
    }
