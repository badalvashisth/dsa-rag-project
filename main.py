# ============================================================
# main.py — Web API entry point for Render deployment
#
# This file does NOT modify ingestion.py or query.py.
# It reuses the embeddings / pinecone_index / llm objects that
# query.py already builds, and exposes the same retrieval +
# generation logic as an HTTP endpoint so a frontend (or your
# luxury UI) can call it, and so Render has something to run
# as a web service (query.py's main() is a blocking input()
# loop, which can't serve as a web server).
# ============================================================

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Safe to import: query.py only starts its input() loop when
# query.py itself is run directly (it's behind
# `if __name__ == "__main__":`). Importing it here just gives
# us the already-configured clients.
from query import embeddings, pinecone_index, llm

app = FastAPI(title="DSA RAG API")

# Allow the frontend to call this API even if it's ever hosted
# on a different domain (e.g. Vercel/Netlify) than the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


def get_answer(question: str) -> str:
    """
    Same logic as query.py's chatting() function, just returning
    the answer instead of printing it to a terminal.
    """
    query_vector = embeddings.embed_query(question)

    search_results = pinecone_index.query(
        top_k=10,
        vector=query_vector,
        include_metadata=True,
    )

    context = "\n\n---\n\n".join(
        match["metadata"]["text"] for match in search_results["matches"]
    )

    prompt = f"""
You are a Data Structure and Algorithm Expert.

You will be given a context of relevant information
and a user question.

Your task is to answer the user's question based ONLY
on the provided context.

If the answer is not in the context, you must say:

"I could not find the answer in the provided document."

Keep your answers clear, concise, and educational.

Context:
{context}

User Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content


@app.post("/api/ask")
def ask(payload: Question):
    answer = get_answer(payload.question)
    return {"answer": answer}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve your frontend: drop your index.html / css / js into a
# folder named "static" next to this file, and it will be
# served at "/". If that folder doesn't exist yet, the API
# still works on its own at /api/ask.
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
