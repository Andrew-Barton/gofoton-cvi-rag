from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY, LLM_MODEL

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
