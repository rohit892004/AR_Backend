import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_PATH = "engine_knowledge.pdf"
VECTOR_DB_PATH = "engine_vector_db"

def create_vector_db():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError("PDF file not found")

    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📦 Saving Vector DB...")
    Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    print("✅ Vector DB created successfully!")

if __name__ == "__main__":
    create_vector_db()
