from db import BengaliVectorStore
from llm import BengaliRAGSystem
from embedder import BengaliEmbedder
from chunker import BengaliChunker
import os

# Initialize components
embedder = BengaliEmbedder()
chunker = BengaliChunker()
vector_db = BengaliVectorStore()
rag = BengaliRAGSystem(vector_db)

# Example document processing pipeline
def process_documents(pdf_path: str):
    # 1. Extract text (from your extractor.py)
    text = extract_text_from_pdf(pdf_path)
    
    # 2. Chunk text
    chunks = chunker.chunk_text(text)
    
    # 3. Embed and store
    embedded_chunks = embedder.embed_chunks(chunks)
    vector_db.create_from_texts(
        texts=[chunk["text"] for chunk in embedded_chunks],
        metadatas=[chunk["metadata"] for chunk in embedded_chunks]
    )
    vector_db.save_local("vector_db")

# Example query
def ask_question(question: str):
    response = rag.ask(question)
    print(f"Q: {question}")
    print(f"A: {response['answer']}")
    print("Sources:")
    for src in response["sources"]:
        print(f"- {src.get('source', 'Unknown')}")

if __name__ == "__main__":
    # Process documents (one-time)
    process_documents("HSC26-Bangla1st-Paper.pdf")
    
    # Interactive Q&A
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == 'quit':
            break
        ask_question(question)