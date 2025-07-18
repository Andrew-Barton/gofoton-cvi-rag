# cli_chat.py

from app.document_loader import load_documents, create_vectorstore
from app.rag_chain import create_qa_chain, normalize_query, add_voice_friendly_ending
from langchain.retrievers import EnsembleRetriever

# TEMP: Scan chunks manually for debug
docs = load_documents("data/sample_docs")

# TEMP DEBUG: Search for 2.78 in the actual chunks
print("\n📦 Searching for chunks containing '2.78'...\n")
for doc in docs:
    if "2.78" in doc:
        print("----- MATCH FOUND -----")
        print(doc)
        print()

def chat_loop():
    print("📡 GoFoton AI Assistant (Type 'exit' to quit)")
    print("Loading documents and setting up...")

    docs = load_documents("data/sample_docs")
    vectorstore = create_vectorstore(docs)

    # Set up base similarity retriever
    similarity = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    # Set up keyword-based retriever (for exact matches like “OD”, “2.78mm”)
    keyword = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

    # Combine them (weights can be tuned)
    retriever = EnsembleRetriever(
        retrievers=[similarity, keyword],
        weights=[0.7, 0.3]
    )

    qa = create_qa_chain(retriever)

    while True:
        query = input("🧠 Ask a question: ")
        if query.strip().lower() == "exit":
            break

        try:
            # 🔁 Step 1: Normalize query
            rewritten = normalize_query(query, qa["llm"])

            # 🔍 Step 2: Get docs and answer
            docs = qa["retriever"].invoke(rewritten)

            response = qa["chain"].invoke({
                "question": rewritten,
                "context": docs  # 👈 Keep this 'context' — LangChain expects this!
            })

            # 🎙️ Step 3: Add voice-friendly ending
            final = add_voice_friendly_ending(response)
            print(f"🤖 Answer:\n{final}\n")

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    chat_loop()
