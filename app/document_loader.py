import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import OPENAI_API_KEY
from langchain.docstore.document import Document
from pdfminer.high_level import extract_text

def load_pdf(path):
    try:
        text = extract_text(path)
        return text
    except Exception as e:
        print(f"❌ Failed to extract text from {path}: {e}")
        return ""

def load_documents(directory="data/sample_docs"):
    docs = []
    for filename in os.listdir(directory):
        print(f"📄 Loading: {filename}")
        if filename.endswith(".pdf"):
            full_path = os.path.join(directory, filename)
            raw_text = load_pdf(full_path)
            docs.append(Document(page_content=raw_text))
        elif filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as f:
                text = f.read()
                docs.append(Document(page_content=text))
    return docs

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=".chroma_store")

    print(f"📦 Created {len(chunks)} document chunks.")
    for i, chunk in enumerate(chunks[:5]):  # Preview first 5
        print(f"\n--- Chunk {i+1} ---\n{chunk.page_content[:500]}")


    return vectorstore
