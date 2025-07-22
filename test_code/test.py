# Modified version of the BanglaRAGChain class to support Gemini 2.5 Flash

import os
import logging
from google import genai
import warnings
from dotenv import load_dotenv
import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Load environment variables from .env file
load_dotenv()

import sys
sys.path.append(r'C:\Users\Mohammad Akash\Documents\projects\bangla_rag')

class BanglaRAGChainWithGemini:

    def __init__(self):
        self.embed_model = None
        self.retriever = None
        self.gemini_client = None
        self.k = 4
        self.temperature = 0.6
        self.max_new_tokens = 256
        
    def load(self, 
             gemini_api_key=None,
             embed_model_id="l3cube-pune/bengali-sentence-similarity-sbert",
             text_path=None,
             k=4,
             temperature=0.6,
             chunk_size=500,
             chunk_overlap=150,
             max_new_tokens=256,
             **kwargs):
        """
        Load the RAG system with Gemini 2.5 Flash as the chat model
        """
        try:
            # Initialize Gemini client
            if gemini_api_key:
                os.environ['GEMINI_API_KEY'] = gemini_api_key
            
            # The client gets the API key from the environment variable `GEMINI_API_KEY`
            self.gemini_client = genai.Client()
            
            # Set parameters
            self.k = k
            self.temperature = temperature
            self.max_new_tokens = max_new_tokens
            
            # Load embedding model (you'll still need HuggingFace for embeddings)
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(embed_model_id)
            
            # Load and process text (simplified version)
            if text_path:
                self._load_and_index_text(text_path, chunk_size, chunk_overlap)
                
            logging.info("RAG system loaded with Gemini 2.5 Flash successfully!")
            
        except Exception as e:
            logging.error(f"Failed to load RAG system: {e}")
            raise
    
    def _load_and_index_text(self, text_path, chunk_size, chunk_overlap):
        """
        Load text file and create vector index
        """
        # Read text file
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple text chunking (you might want to use more sophisticated chunking)
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        
        # Create embeddings
        self.chunks = chunks
        self.chunk_embeddings = self.embed_model.encode(chunks)
        
        logging.info(f"Indexed {len(chunks)} text chunks")
    
    def _retrieve_relevant_chunks(self, query):
        """
        Retrieve most relevant chunks for the query
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Encode query
        query_embedding = self.embed_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top k chunks
        top_indices = np.argsort(similarities)[-self.k:][::-1]
        relevant_chunks = [self.chunks[i] for i in top_indices]
        
        return relevant_chunks
    
    def get_response(self, query):
        """
        Generate response using Gemini 2.5 Flash
        """
        try:
            # Retrieve relevant context
            relevant_chunks = self._retrieve_relevant_chunks(query)
            context = "\n\n".join(relevant_chunks)
            
            # Create prompt for Gemini
            prompt = f"""আপনি একটি বাংলা প্রশ্ন-উত্তর সহায়ক। নিচের প্রসঙ্গ ব্যবহার করে প্রশ্নের উত্তর দিন।

প্রসঙ্গ:
{context}

প্রশ্ন: {query}

অনুগ্রহ করে বাংলায় সংক্ষিপ্ত এবং সঠিক উত্তর দিন। যদি প্রসঙ্গে উত্তর না থাকে, তাহলে "আমি এই প্রশ্নের উত্তর প্রদত্ত প্রসঙ্গে খুঁজে পাইনি" বলুন।

উত্তর:"""

            # Generate response using Gemini 2.5 Flash
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt
            )
            
            answer = response.text.strip()
            
            return answer, context
            
        except Exception as e:
            logging.error(f"Failed to generate response: {e}")
            raise

# Modified main function
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bangla RAG System with Gemini 2.5 Flash"
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        help="Your Google AI Studio API key (optional if set in .env file)",
        default=os.getenv('GEMINI_API_KEY')
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="l3cube-pune/bengali-sentence-similarity-sbert",
        help="The Hugging Face model ID of the embedding model",
    )
    parser.add_argument(
        "--k", type=int, default=4, help="The number of documents to retrieve"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The temperature parameter for Gemini",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="The maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="The chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=150,
        help="The chunk overlap for text splitting",
    )
    parser.add_argument(
        "--text_path",
        type=str,
        required=True,
        help="The txt file path to the text file",
    )
    parser.add_argument(
        "--show_context",
        action="store_true",
        help="Whether to show the retrieved context or not.",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="The question to ask the RAG system.",
    )
    
    args = parser.parse_args()
    
    # Check if API key is available
    if not args.gemini_api_key:
        print("Error: GEMINI_API_KEY not found in .env file or command line arguments!")
        print("Please either:")
        print("1. Add GEMINI_API_KEY=your_key to your .env file")
        print("2. Use --gemini_api_key your_key argument")
        return
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Initialize RAG system with Gemini
        rag_chain = BanglaRAGChainWithGemini()
        rag_chain.load(
            gemini_api_key=args.gemini_api_key,
            embed_model_id=args.embed_model,
            text_path=args.text_path,
            k=args.k,
            temperature=args.temperature,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_new_tokens=args.max_new_tokens,
        )
        
        # Get the question from the command-line arguments
        query = args.question

        if query:
            try:
                answer, context = rag_chain.get_response(query)
                if args.show_context:
                    print(f"প্রসঙ্গঃ {context}\n------------------------\n")
                print(f"উত্তর: {answer}")
            except Exception as e:
                logging.error(f"Couldn't generate an answer: {e}")
                print("আবার চেষ্টা করুন!")
        else:
            # Interactive loop
            print("বাংলা RAG সিস্টেম Gemini 2.5 Flash সহ প্রস্তুত!")
            while True:
                query = input("আপনার প্রশ্ন: ")
                if query.lower() in ["exit", "quit"]:
                    print("আবার দেখা হবে, ধন্যবাদ!")
                    break
                try:
                    answer, context = rag_chain.get_response(query)
                    if args.show_context:
                        print(f"প্রসঙ্গঃ {context}\n------------------------\n")
                    print(f"উত্তর: {answer}")
                except Exception as e:
                    logging.error(f"Couldn't generate an answer: {e}")
                    print("আবার চেষ্টা করুন!")
                
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        print("Error occurred, please check logs for details.")

if __name__ == "__main__":
    main()