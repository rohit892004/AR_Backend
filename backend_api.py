import os
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

from create_vector_db import create_vector_db

# ---------------- CONFIG ----------------
VECTOR_DB_PATH = "/data/engine_vector_db"   # Render Disk path
PDF_PATH = "engine_knowledge.pdf"

# ---------------------------------------
app = FastAPI(
    title="AR Engine AI Backend",
    version="1.0"
)

# ---------------- MODEL ----------------
class QueryRequest(BaseModel):
    query: str

# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup_event():
    if not os.path.exists(VECTOR_DB_PATH):
        print("Vector DB not found. Creating...")
        create_vector_db(VECTOR_DB_PATH)
    else:
        print("Vector DB found.")

# ---------------- LOAD AI ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

llm = Ollama(model="llama3")  # local / server LLM

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

# ---------------- API ----------------
@app.post("/ask")
def ask_engine(data: QueryRequest):
    result = qa_chain.run(data.query)
    return {"answer": result}
