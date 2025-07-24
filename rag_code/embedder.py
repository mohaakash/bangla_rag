from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict
import numpy as np
import torch
import re
import warnings

class BengaliEmbedder:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model = self._initialize_model(model_name, device)
        self.embedding_dim = self._get_embedding_dim()
        
    def _initialize_model(self, model_name: str, device: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            
            return HuggingFaceEmbeddings(
                model_name=model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': device},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32 if device == "cpu" else 128
                }
            )

    def _get_embedding_dim(self) -> int:
        dummy_text = "test"
        return len(self.model.embed_query(dummy_text))

    # Preprocess text to standardize Bengali punctuation and whitespace
    # This is a simplified version, you can expand it based on your needs
    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'[।]+', '।', text)  # Multiple dandas to single
        text = re.sub(r'[\s]+', ' ', text).strip()
        return text

    # Embed a list of text chunks with metadata
    # Each chunk should be a dict with 'text' and 'metadata' keys
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []

        processed_texts = [self.preprocess_text(chunk["text"]) for chunk in chunks]
        
        # Batch embed all texts
        embeddings = self.model.embed_documents(processed_texts)
        
        # Combine with original metadata
        return [{
            "text": chunk["text"],
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            "metadata": {
                **chunk["metadata"],
                "embedding_dim": self.embedding_dim,
                "is_normalized": True
            }
        } for chunk, embedding in zip(chunks, embeddings)]

