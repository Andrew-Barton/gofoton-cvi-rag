from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from app.config import OPENAI_API_KEY, LLM_MODEL
from typing import List
from pydantic import PrivateAttr


class LoggingRetriever(BaseRetriever):
    _retriever: any = PrivateAttr()

    def __init__(self, retriever):
        super().__init__()
        self._retriever = retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._retriever.get_relevant_documents(query)


def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)

    # Semantic reranking retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    reranker = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    retriever = LoggingRetriever(reranker)

    # ✅ Prompt expects `context` and `question`
    prompt = PromptTemplate(
        template="""
You are a helpful assistant representing GoFoton.
Respond in a natural, professional tone, as if you're speaking to a customer.
Answer clearly and concisely based on the provided information.

Question: {question}
=========
{context}
=========
Helpful Answer:""",
        input_variables=["context", "question"]
    )

    # 🧠 Manually construct chain from LLM and prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    return RetrievalQAWithSourcesChain(
        retriever=retriever,
        combine_documents_chain=stuff_chain,
        return_source_documents=True
    )
