import os
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chains import RetrievalQA

from create_vector_db import create_vector_db

# ---------------- CONFIG ----------------
VECTOR_DB_PATH = "/data/engine_vector_db"   # Render disk mount
PDF_PATH = "engine_knowledge.pdf"

# --------------------------------------
app = FastAPI(
    title="AR Engine AI Backend",
    version="1.0"
)

# ---------------- REQUEST MODEL ----------------
class QueryRequest(BaseModel):
    query: str

# ---------------- STARTUP EVENT ----------------
@app.on_event("startup")
def startup_event():
    if not os.path.exists(VECTOR_DB_PATH):
        print("⚠️ Vector DB not found. Creating now...")
        create_vector_db(VECTOR_DB_PATH)
    else:
        print("✅ Vector DB already exists.")

# ---------------- LOAD VECTOR DB ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# ---------------- LLM (OLLAMA) ----------------
llm = Ollama(model="llama3")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=False
)

# ---------------- API ENDPOINT ----------------
@app.post("/ask")
def ask_engine(data: QueryRequest):
    answer = qa_chain.run(data.query)
    return {
        "answer": answer
    }
