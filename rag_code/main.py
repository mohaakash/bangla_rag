# Enhanced main file for Bengali RAG system
from db import BengaliVectorStore
from llm import BengaliRAGSystem
from embedder import BengaliEmbedder
from chunker import BengaliChunker
from extractor import extract_text_from_pdf
import os
import time

# --- Configuration ---
PDF_PATH = "HSC26-Bangla1st-Paper.pdf"
VECTOR_DB_PATH = "bengali_vector_db"

# Enhanced chunker configuration for Bengali text
CHUNKER_CONFIG = {
    "max_chunk_size": 400,  # Slightly smaller for better semantic coherence
    "overlap": 80,          # Increased overlap for better context preservation
    "min_chunk_size": 50,   # Minimum viable chunk size
}

# --- Utility Functions ---
def print_chunking_stats(chunks):
    """Print statistics about the chunking process"""
    if not chunks:
        print("No chunks generated.")
        return
    
    total_chunks = len(chunks)
    avg_length = sum(len(chunk["text"]) for chunk in chunks) / total_chunks
    
    # Language distribution
    lang_counts = {}
    discourse_marker_count = 0
    
    for chunk in chunks:
        lang = chunk["metadata"]["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        if chunk["metadata"]["has_discourse_markers"]:
            discourse_marker_count += 1
    
    print(f"\n--- Chunking Statistics ---")
    print(f"Total chunks: {total_chunks}")
    print(f"Average chunk length: {avg_length:.1f} characters")
    print(f"Language distribution: {lang_counts}")
    print(f"Chunks with discourse markers: {discourse_marker_count}")
    
    # Show first few chunks for verification
    print(f"\n--- Sample Chunks ---")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"Chunk {i} ({chunk['metadata']['language']}):")
        preview = chunk["text"][:150] + "..." if len(chunk["text"]) > 150 else chunk["text"]
        print(f"  {preview}")
        print(f"  Bengali: {chunk['metadata']['bengali_ratio']:.2f}, "
              f"English: {chunk['metadata']['english_ratio']:.2f}")
        print()

def validate_chunks(chunks):
    """Validate chunk quality and provide warnings"""
    issues = []
    
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        metadata = chunk["metadata"]
        
        # Check for very short chunks
        if len(text) < 30:
            issues.append(f"Chunk {i+1}: Very short ({len(text)} chars)")
        
        # Check for very long chunks without sentence endings
        if len(text) > 500 and not metadata["has_sentence_end"]:
            issues.append(f"Chunk {i+1}: Long chunk without proper sentence ending")
        
        # Check for chunks with no meaningful content
        if len(text.strip()) < 10:
            issues.append(f"Chunk {i+1}: Potentially empty or meaningless content")
    
    if issues:
        print(f"\n--- Chunking Issues Found ---")
        for issue in issues[:5]:  # Show max 5 issues
            print(f"  {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")
    else:
        print("✓ Chunk validation passed")

# --- Initialization ---
def initialize_rag_system():
    """Initializes all components of the RAG system with enhanced chunking."""
    print("Initializing enhanced Bengali RAG system...")
    
    embedder = BengaliEmbedder()
    
    # Initialize enhanced chunker with optimized settings
    chunker = BengaliChunker(**CHUNKER_CONFIG)
    
    vector_db = BengaliVectorStore()
    
    # Check if the vector database and its index file exist
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(os.path.join(VECTOR_DB_PATH, "index.pkl")):
        print(f"Loading existing vector database from {VECTOR_DB_PATH}")
        vector_db.load_local(VECTOR_DB_PATH)
        print("✓ Vector database loaded successfully")
    else:
        # Create the directory if it doesn't exist
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        print(f"No vector database found. Processing new PDF: {PDF_PATH}")
        
        # 1. Extract text from the PDF
        print("Extracting text from PDF...")
        start_time = time.time()
        text = extract_text_from_pdf(PDF_PATH)
        
        if not text.strip():
            print("Error: No text could be extracted from the PDF. Please check the file.")
            return None
        
        print(f"✓ Extracted {len(text)} characters in {time.time() - start_time:.2f}s")

        # 2. Enhanced chunking with statistics
        print("Chunking text with enhanced Bengali chunker...")
        start_time = time.time()
        chunks = chunker.chunk_text(text)
        print(f"✓ Chunking completed in {time.time() - start_time:.2f}s")
        
        # Display chunking statistics
        print_chunking_stats(chunks)
        
        # Validate chunk quality
        validate_chunks(chunks)

        if not chunks:
            print("Error: No chunks were generated from the text.")
            return None

        # 3. Embed the chunks and create the vector store
        print("Embedding chunks and creating vector store...")
        start_time = time.time()
        
        try:
            vector_db.create_from_texts(
                texts=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks]
            )
            print(f"✓ Vector store created in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None
        
        # 4. Save the vector store for future use
        print(f"Saving vector database to {VECTOR_DB_PATH}")
        try:
            vector_db.save_local(VECTOR_DB_PATH)
            print("✓ Vector database saved successfully")
        except Exception as e:
            print(f"Warning: Could not save vector database: {e}")

    # 5. Initialize the RAG system with the vector store
    try:
        rag_system = BengaliRAGSystem(vector_db)
        print("✓ RAG system is ready")
        return rag_system
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return None

# --- Main Application Loop ---
def main():
    """Main function to run the interactive RAG chat with enhanced features."""
    print("=== Enhanced Bengali RAG System ===")
    print("Features: Improved script mixing, word-boundary overlap, discourse markers")
    print()
    
    rag = initialize_rag_system()
    if not rag:
        print("Failed to initialize RAG system. Please check your configuration.")
        return

    print("\n" + "="*50)
    print("--- Bengali RAG is now running ---")
    print("Features:")
    print("  • Mixed Bengali-English text support")
    print("  • Smart discourse marker recognition")
    print("  • Word-boundary preserving overlap")
    print("  • Enhanced semantic chunking")
    print()
    print("Commands:")
    print("  • Type your question in Bengali or English")
    print("  • Type 'stats' to see system statistics")
    print("  • Type 'help' for more commands")
    print("  • Type 'quit' or 'exit' to stop")
    print("="*50)

    question_count = 0
    
    while True:
        try:
            question = input("\n🤔 Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'বন্ধ']:
                print("👋 Exiting the program. Goodbye! / বিদায়!")
                break
            
            if not question:
                continue
            
            # Handle special commands
            if question.lower() == 'help':
                print("\n📚 Available commands:")
                print("  • Ask any question in Bengali or English")
                print("  • 'stats' - Show system statistics")
                print("  • 'config' - Show current configuration")
                print("  • 'help' - Show this help message")
                print("  • 'quit'/'exit' - Exit the program")
                continue
            
            if question.lower() == 'stats':
                print(f"\n📊 System Statistics:")
                print(f"  • Questions answered: {question_count}")
                print(f"  • Chunker config: {CHUNKER_CONFIG}")
                print(f"  • Vector DB path: {VECTOR_DB_PATH}")
                continue
            
            if question.lower() == 'config':
                print(f"\n⚙️ Current Configuration:")
                for key, value in CHUNKER_CONFIG.items():
                    print(f"  • {key}: {value}")
                continue

            print("\n🧠 Thinking...")
            start_time = time.time()
            
            try:
                response = rag.ask(question)
                response_time = time.time() - start_time
                question_count += 1
                
                print(f"\n💡 Answer (responded in {response_time:.2f}s):")
                print("-" * 40)
                print(response['answer'])
                print("-" * 40)
                
                # Optional: Show source information
                if 'sources' in response and response['sources']:
                    print(f"\n📄 Sources ({len(response['sources'])} chunks used):")
                    for i, src in enumerate(response["sources"][:3], 1):  # Show max 3 sources
                        lang_info = f"{src.get('language', 'N/A')}"
                        if 'bengali_ratio' in src:
                            lang_info += f" (BN:{src['bengali_ratio']:.1f}, EN:{src['english_ratio']:.1f})"
                        print(f"  [{i}] {lang_info}, {src.get('length_chars', 'N/A')} chars")

            except Exception as e:
                print(f"❌ Error processing question: {e}")
                print("Please try rephrasing your question.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    # Set up the Google API key from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it to your Google Generative AI API key.")
        print("\nExample:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
    else:
        main()