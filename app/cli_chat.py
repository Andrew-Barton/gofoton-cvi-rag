from app.document_loader import load_documents, create_vectorstore
from app.rag_chain import create_qa_chain
import textwrap

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
        answer = result['answer'].replace("  ", " ").strip()
        wrapped = textwrap.fill(answer, width=80)
        print(f"🤖 Answer:\n{wrapped}")
        if result.get("sources"):
            print(f"📚 Sources: {result['sources']}")

if __name__ == "__main__":
    chat_loop()
