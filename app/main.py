from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.document_loader import load_documents, create_vectorstore
from app.rag_chain import create_qa_chain

app = FastAPI()

# Allow CORS (so frontend can talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load everything once at startup
print("🔄 Loading vector store and QA chain...")
docs = load_documents()
try:
    vectorstore = create_vectorstore(docs)
    qa_chain = create_qa_chain(vectorstore)
    print("✅ RAG backend ready.")
except Exception as e:
    print(f"❌ Error initializing backend: {e}")

qa_chain = create_qa_chain(vectorstore)
print("✅ RAG backend ready.")

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(payload: Question):
    answer = qa_chain.invoke({"question": payload.question})
    return {"answer": answer["result"]}

@app.get("/")
def health_check():
    return {"status": "GoFoton RAG backend is live"}
