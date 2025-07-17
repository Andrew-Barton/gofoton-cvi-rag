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
        full_path = os.path.join(directory, filename)

        if filename.endswith(".pdf"):
            raw_text = load_pdf(full_path)
        elif filename.endswith(".txt"):
            with open(full_path, "r") as f:
                raw_text = f.read()
        else:
            continue

        docs.append(Document(page_content=raw_text, metadata={"source": filename}))
    return docs

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"📦 Split into {len(chunks)} chunks")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=".chroma_store")
    return vectorstore
