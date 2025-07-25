from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from .db import BengaliVectorStore
import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()  # load from .env

class BengaliRAGSystem:
    def __init__(self, vector_store: BengaliVectorStore):
        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Configure memory (last 3 exchanges)
        self.memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Bilingual prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        [System] Answer in the same language as the question.Do not explain the answer keep things short.
        Use the following context and conversation history:
        
        প্রসঙ্গ/Context:
        {context}
        
        আলাপচারিতা/Conversation:
        {chat_history}
        
        প্রশ্ন/Question: {question}
        উত্তর/Answer:""")
        
        # Create retrieval chain
        self.retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(k=8),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            verbose=True
        )

    def ask(self, question: str) -> Dict:
        """Process question with context and history"""
        result = self.retrieval_chain.invoke({"question": question})
        return {
            "answer": result["answer"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }