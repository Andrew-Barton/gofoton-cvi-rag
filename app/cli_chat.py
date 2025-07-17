from app.document_loader import load_documents, create_vectorstore
from app.rag_chain import create_qa_chain

def chat_loop():
    print("📡 GoFoton AI Assistant (Type 'exit' to quit)")
    print("Loading documents and setting up...")

    docs = load_documents()
    vectorstore = create_vectorstore(docs)
    qa = create_qa_chain(vectorstore)

    while True:
        query = input("🧠 Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa.invoke({"question": query})
        print(f"🤖 Answer: {result['answer']}")
        if result.get("sources"):
            print(f"📚 Sources: {result['sources']}")


if __name__ == "__main__":
    chat_loop()
