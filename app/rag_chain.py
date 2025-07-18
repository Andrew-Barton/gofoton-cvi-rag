# rag_chain.py

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
import os

# 1. Setup LLM
def get_llm():
    return ChatOpenAI(
        model="gpt-4",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )

# 2. Synonym Normalization Prompt
rephrase_prompt = PromptTemplate.from_template("""
You are a helpful assistant optimizing user queries for better understanding.
Rewrite the following user query using more standard and descriptive terminology:
"{query}"
""")

def normalize_query(query: str, llm: ChatOpenAI) -> str:
    rephraser_chain = rephrase_prompt | llm | StrOutputParser()
    return rephraser_chain.invoke({"query": query}).strip()

# 3. Voice CTA
def add_voice_friendly_ending(response: str) -> str:
    if "I don't know" in response.lower() or "not available" in response.lower():
        return response
    return f"{response}\n\nLet me know if you'd like more details — I'm happy to help."

# 4. Prompt for document answering
rag_prompt = PromptTemplate.from_template("""
Use the following context, which may include technical specifications or tabular data, to answer the question clearly and helpfully.

{context}

Question:
{question}

Helpful Answer:"""
)

# 5. Main QA Chain Factory
def create_qa_chain(retriever):
    llm = get_llm()

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=rag_prompt
    )

    return {
        "llm": llm,
        "retriever": retriever,
        "chain": chain
    }
