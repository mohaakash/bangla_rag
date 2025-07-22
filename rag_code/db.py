import uuid
import chromadb
from chromadb.utils import embedding_functions
from chunker import chunk_text_for_gemini as chunk_text

def get_embedding_function(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Returns a Chroma-compatible embedding function using sentence-transformers.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

def create_chroma_collection(
    collection_name="bangla_rag",
    persist_dir="chroma_store",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    """
    Initializes Chroma DB with a persistent collection.
    If it exists, returns the existing collection.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = get_embedding_function(model_name)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    return collection

def insert_text_to_chroma(text, collection, source="bangla_pdf", max_tokens=500, overlap_tokens=50, language="bengali"):
    """
    Chunks text and inserts them into ChromaDB with rich metadata.

    Parameters:
    - text: The raw text to be chunked and inserted.
    - collection: Chroma collection object
    - source: metadata label for source (default: 'bangla_pdf')
    - max_tokens: Maximum tokens per chunk.
    - overlap_tokens: Number of tokens to overlap between chunks.
    - language: Language for sentence tokenization.
    """
    chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadata_list = []

    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": i,
            "source": source
        }
        # Page numbers are not directly available from raw text, so we omit them here.
        # If page numbers are needed, the text extraction process needs to provide them per sentence/paragraph.
        metadata_list.append(metadata)

    collection.add(documents=chunks, metadatas=metadata_list, ids=ids)
    print(f"✅ Inserted {len(chunks)} chunks with metadata into ChromaDB.")

def query_chroma(collection, query_text, n_results=3):
    """
    Queries the ChromaDB collection for top N relevant chunks.

    Returns:
    - Dictionary with 'documents', 'metadatas', etc.
    """
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results
