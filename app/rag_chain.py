from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import PrivateAttr
from app.config import OPENAI_API_KEY, LLM_MODEL

class LoggingRetriever(BaseRetriever):
    _retriever: any = PrivateAttr()

    def __init__(self, retriever):
        super().__init__()
        self._retriever = retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        print(f"\n🧪 get_relevant_documents() called with query: {query}")
        docs = self._retriever.get_relevant_documents(query)
        print(f"\n🔎 Retrieved {len(docs)} docs for: '{query}'")
        for i, doc in enumerate(docs[:3]):
            print(f"\n--- Doc {i+1} ---\n{doc.page_content[:300]}")
        return docs

def create_qa_chain(vectorstore):
    base_retriever = vectorstore.as_retriever()
    wrapped_retriever = LoggingRetriever(base_retriever)
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
    return RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=wrapped_retriever,
        return_source_documents=True
    )
