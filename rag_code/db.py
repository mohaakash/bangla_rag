from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict
import os

class BengaliVectorStore:
    def __init__(self, embedding_model="paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None

    def create_from_texts(self, texts: List[str], metadatas: List[Dict] = None):
        """Create vectorstore from texts"""
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embedding,
            metadatas=metadatas
        )
        return self

    def save_local(self, path: str):
        """Save vectorstore to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_local(self, path: str):
        """Load existing vectorstore"""
        self.vectorstore = FAISS.load_local(
            folder_path=path,
            embeddings=self.embedding,
            allow_dangerous_deserialization=True
        )
        return self

    def as_retriever(self, k: int = 4, language_filter: str = None):
        """Create language-aware retriever"""
        def _language_filter(doc):
            if not language_filter:
                return True
            return doc.metadata.get("language") == language_filter

        return self.vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": _language_filter,
                "score_threshold": 0.65
            }
        )