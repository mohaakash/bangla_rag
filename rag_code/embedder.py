from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict
import numpy as np
import torch
import re
import warnings

class BengaliEmbedder:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        """
        Initialize embedding model optimized for Bengali-English content
        
        Args:
            model_name: Optional custom model name
            device: "cpu" or "cuda"
        """
        self.model = self._initialize_model(model_name, device)
        self.embedding_dim = self._get_embedding_dim()
        
    def _initialize_model(self, model_name: str, device: str):
        """Load best available multilingual embedding model"""
        # Suppress specific LangChain deprecation warnings
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
        """Get embedding dimension size for database configuration"""
        dummy_text = "test"
        return len(self.model.embed_query(dummy_text))

    def preprocess_text(self, text: str) -> str:
        """Normalize Bengali-English text before embedding"""
        # Standardize Bengali punctuation
        text = re.sub(r'[।]+', '।', text)  # Multiple dandas to single
        # Normalize whitespace and special characters
        text = re.sub(r'[\s]+', ' ', text).strip()
        return text

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Embed text chunks with metadata preservation
        
        Args:
            chunks: List of chunk dicts from chunker.py
                   Format: [{"text": "...", "metadata": {...}}, ...]
        
        Returns:
            List of dicts with embeddings and metadata
            Format: [{"text": "...", "embedding": [...], "metadata": {...}}, ...]
        """
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

if __name__ == "__main__":
    # Test with sample chunks (matching chunker.py output format)
    sample_chunks = [
        {
            "text": "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ।",
            "metadata": {"language": "bn", "length": 34}
        },
        {
            "text": "The capital is Dhaka.",
            "metadata": {"language": "en", "length": 18}
        }
    ]
    
    # Initialize with automatic device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = BengaliEnglishEmbedder(device=device)
    
    embedded_chunks = embedder.embed_chunks(sample_chunks)
    
    print(f"Using device: {device}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    print(f"First chunk embedding (sample): {embedded_chunks[0]['embedding'][:5]}...")