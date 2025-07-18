import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import OPENAI_API_KEY
from langchain.docstore.document import Document
from pdfminer.high_level import extract_text
from langchain_core.documents import Document
from pathlib import Path
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import tempfile
from typing import List
from langchain_community.document_loaders import TextLoader


def ocr_fallback_loader(pdf_path: str, min_length_threshold: int = 100) -> list[Document]:
    """
    Attempt to extract text from a PDF. If too little is returned, fallback to OCR.
    """
    print(f"📄 Attempting text extraction: {Path(pdf_path).name}")
    extracted_text = extract_text(pdf_path)
    
    if extracted_text and len(extracted_text.strip()) >= min_length_threshold:
        return [Document(page_content=extracted_text, metadata={"source": Path(pdf_path).name})]

    print("⚠️ Text extraction failed or insufficient. Using OCR fallback...")
    with tempfile.TemporaryDirectory() as tmpdir:
        images = convert_from_path(pdf_path, output_folder=tmpdir)
        ocr_text = ""
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            ocr_text += f"\n\n--- Page {i+1} ---\n{page_text.strip()}"
    
    return [Document(page_content=ocr_text, metadata={"source": Path(pdf_path).name})]

def load_pdf(path):
    try:
        text = extract_text(path)
        return text
    except Exception as e:
        print(f"❌ Failed to extract text from {path}: {e}")
        return ""

def load_documents(directory: str = "data/sample_docs") -> List[Document]:
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if filename.endswith(".txt"):
            print(f"📄 Loading: {filename}")
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata["loader"] = "text"
            documents.extend(docs)

        elif filename.endswith(".pdf"):
            print(f"📄 Loading: {filename}")
            try:
                text = extract_text(filepath)
                if not text.strip():
                    raise ValueError("Empty PDF content — falling back to OCR")
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "loader": "pdf-native"}
                ))
            except Exception as e:
                print(f"⚠️ PDF extraction failed for {filename}, using OCR fallback: {e}")
                ocr_docs = ocr_fallback_loader(filepath, filename)
                for d in ocr_docs:
                    d.metadata["loader"] = "pdf-ocr"
                documents.extend(ocr_docs)

        # Placeholder for future support
        # elif filename.endswith(".docx"):
        #     pass  # e.g., use python-docx
        # elif filename.endswith(".rtf"):
        #     pass  # rtf-parser
        # elif filename.endswith((".jpg", ".jpeg", ".png")):
        #     documents.extend(ocr_fallback_loader(filepath, filename))

    return documents

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"📦 Split into {len(chunks)} chunks")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=".chroma_store")
    return vectorstore
