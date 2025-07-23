from db import BengaliVectorStore
from llm import BengaliRAGSystem
from embedder import BengaliEmbedder
from chunker import BengaliChunker
from extractor import extract_text_from_pdf
import os

# --- Configuration ---
PDF_PATH = "HSC26-Bangla1st-Paper.pdf"
VECTOR_DB_PATH = "bengali_vector_db"

# --- Initialization ---
def initialize_rag_system():
    """Initializes all components of the RAG system."""
    print("Initializing RAG system...")
    embedder = BengaliEmbedder()
    chunker = BengaliChunker()
    vector_db = BengaliVectorStore()
    
    # Check if the vector database and its index file exist
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(os.path.join(VECTOR_DB_PATH, "index.pkl")):
        print(f"Loading existing vector database from {VECTOR_DB_PATH}")
        vector_db.load_local(VECTOR_DB_PATH)
    else:
        # Create the directory if it doesn't exist
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        print(f"No vector database found. Processing new PDF: {PDF_PATH}")
        # 1. Extract text from the PDF
        text = extract_text_from_pdf(PDF_PATH)
        if not text.strip():
            print("Error: No text could be extracted from the PDF. Please check the file.")
            return None

        # 2. Chunk the extracted text
        print("Chunking text...")
        chunks = chunker.chunk_text(text)

        # 3. Embed the chunks and create the vector store
        print("Embedding chunks and creating vector store...")
        vector_db.create_from_texts(
            texts=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks]
        )
        
        # 4. Save the vector store for future use
        print(f"Saving vector database to {VECTOR_DB_PATH}")
        vector_db.save_local(VECTOR_DB_PATH)

    # 5. Initialize the RAG system with the vector store
    rag_system = BengaliRAGSystem(vector_db)
    print("RAG system is ready.")
    return rag_system

# --- Main Application Loop ---
def main():
    """Main function to run the interactive RAG chat."""
    rag = initialize_rag_system()
    if not rag:
        return

    print("\n--- Bengali RAG is now running ---")
    print("Type your question in Bengali or English.")
    print("Type 'quit' or 'exit' to stop the program.")

    while True:
        try:
            question = input("\nAsk a question: ")
            if question.lower() in ['quit', 'exit']:
                print("Exiting the program. Goodbye!")
                break
            
            if not question.strip():
                continue

            print("\nThinking...")
            response = rag.ask(question)
            
            print("\nAnswer:")
            print(response['answer'])
            
            # Optional: Print sources for transparency
            # print("\nSources:")
            # for i, src in enumerate(response["sources"]):
            #     print(f"  [{i+1}] Language: {src.get('language', 'N/A')}, Length: {src.get('length_chars', 'N/A')}")

        except KeyboardInterrupt:
            print("\nExiting the program. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    # Set up the Google API key from environment variables
    # Make sure to set this in your environment before running
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it to your Google Generative AI API key.")
    else:
        main()