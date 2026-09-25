from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
env_path = r"C:\Users\vashi\OneDrive\Desktop\RAG Project\.env"
load_dotenv(dotenv_path=env_path, override=True)

print("PINECONE_API_KEY loaded:", "YES" if os.getenv("PINECONE_API_KEY") else "NO")
print("PINECONE_INDEX_NAME loaded:", "YES" if os.getenv("PINECONE_INDEX_NAME") else "NO")
print("MISTRAL_API_KEY loaded:", "YES" if os.getenv("MISTRAL_API_KEY") else "NO")

# ------------------------------------------------------------
# STEP 1: Load the PDF
# ------------------------------------------------------------
PDF_PATH = r"C:\Users\vashi\OneDrive\Desktop\RAG Project\dsa.pdf"

pdf_loader = PyPDFLoader(PDF_PATH)
raw_docs = pdf_loader.load()
print(f"✅ Total pages loaded: {len(raw_docs)}")

# ------------------------------------------------------------
# STEP 2: Chunking
# ------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunked_docs = text_splitter.split_documents(raw_docs)
print(f"✅ Total chunks: {len(chunked_docs)}")

# ------------------------------------------------------------
# STEP 3: Embedding model (MISTRAL)
# ------------------------------------------------------------
embeddings = MistralAIEmbeddings(model="mistral-embed")

# ------------------------------------------------------------
# STEP 4: Connect to Pinecone
# ------------------------------------------------------------
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

# ------------------------------------------------------------
# STEP 5: Embed chunks and upload to Pinecone
# ------------------------------------------------------------
vector_store = PineconeVectorStore.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    index_name=os.getenv("PINECONE_INDEX_NAME")
)

print("✅ All chunks uploaded to Pinecone successfully!")
print("✅ Run query.py now to ask questions.")