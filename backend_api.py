import os
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from create_vector_db import create_vector_db

# ---------------- CONFIG ----------------
VECTOR_DB_PATH = "/data/engine_vector_db"

# ---------------- APP ----------------
app = FastAPI(title="AR Engine AI Backend")

# ---------------- REQUEST MODEL ----------------
class QueryRequest(BaseModel):
    query: str

# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup():
    if not os.path.exists(VECTOR_DB_PATH):
        print("Creating Vector DB...")
        create_vector_db(VECTOR_DB_PATH)

# ---------------- VECTOR STORE ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

# ---------------- LLM ----------------
llm = Ollama(model="llama3")

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_template(
    """
Answer the question ONLY using the context below.

<context>
{context}
</context>

Question: {input}
"""
)

doc_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, doc_chain)

# ---------------- API ----------------
@app.post("/ask")
def ask_engine(data: QueryRequest):
    result = qa_chain.invoke({"input": data.query})
    return {"answer": result["answer"]}
