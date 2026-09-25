# ============================================================
# PHASE 2: QUERY RESOLVING
# ============================================================

import os
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_mistralai import MistralAIEmbeddings
from langchain_groq import ChatGroq

# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
env_path = r"C:\Users\vashi\OneDrive\Desktop\RAG Project\.env"
load_dotenv(dotenv_path=env_path, override=True)

# ------------------------------------------------------------
# Embedding model — MISTRAL (same as upload, DO NOT CHANGE)
# ------------------------------------------------------------
embeddings = MistralAIEmbeddings(model="mistral-embed")

# ------------------------------------------------------------
# Pinecone connection
# ------------------------------------------------------------
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

# ------------------------------------------------------------
# Chat LLM — GROQ (replaces Mistral chat)
# ------------------------------------------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# ============================================================
# CHAT FUNCTION
# ============================================================
def chatting(question):

    # 1. Query → vector
    query_vector = embeddings.embed_query(question)

    # 2. Search Pinecone for relevant chunks
    search_results = pinecone_index.query(
        top_k=10,
        vector=query_vector,
        include_metadata=True
    )

    # 3. Extract text from matched chunks
    context = "\n\n---\n\n".join(
        match["metadata"]["text"]
        for match in search_results["matches"]
    )

    # 4. Build prompt
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

    # 5. Ask the LLM
    response = llm.invoke(prompt)

    # 6. Print the answer
    print("\n" + response.content + "\n")


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    print("RAG system ready. Type 'exit' or 'quit' to stop.\n")
    while True:
        question = input("Ask me anything --> ")

        if question.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        chatting(question)


# ============================================================
# START
# ============================================================
if __name__ == "__main__":
    main()