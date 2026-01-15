import os
from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# 🔹 Vector DB path
VECTOR_DB_PATH = "engine_vector_db"

# 🔹 Auto-create Vector DB if missing (RENDER FIX)
if not os.path.exists(VECTOR_DB_PATH):
    print("📦 Vector DB not found. Creating...")
    from create_vector_db import create_vector_db
    create_vector_db()
    print("✅ Vector DB created")

# 🔹 FastAPI app
app = FastAPI(title="AR Backend API")

# 🔹 Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 🔹 Load Vector DB
vector_db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 🔹 LLM
llm_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 🔹 QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# 🔹 Routes
@app.get("/")
def root():
    return {"status": "AR Backend Running 🚀"}

@app.post("/ask")
def ask(question: str):
    return {
        "question": question,
        "answer": qa_chain.run(question)
    }
