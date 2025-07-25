# Bangla RAG System

This project is a Retrieval-Augmented Generation (RAG) system specifically designed for processing and querying documents in Bengali and English. It uses a sophisticated pipeline to extract text from PDF files, clean and chunk the text, generate embeddings, and use a Large Language Model (LLM) to answer questions based on the document content.

## Setup Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mohaakash/bangla_rag.git
    cd bangla_rag
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Tesseract OCR:**
    This project uses Tesseract for Optical Character Recognition. You need to install it on your system and provide the path to the executable in `rag_code/extractor.py`.

    -   **Windows:** Download and install from the official [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki). Update the path in `rag_code/extractor.py`:
        ```python
        pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
        ```
    -   **Linux (Ubuntu/Debian):**
        ```bash
        sudo apt-get install tesseract-ocr
        sudo apt-get install tesseract-ocr-ben
        ```
    -   **macOS (using Homebrew):**
        ```bash
        brew install tesseract
        brew install tesseract-lang
        ```

5.  **Set up your environment variables:**
    Create a `.env` file in the root directory and add your Google API key:
    ```
    GOOGLE_API_KEY="your_google_api_key"
    ```

6.  **Run the application:**
    The main application logic is in `app.py`. You can run it using Streamlit:
    ```bash
    streamlit run app.py
    ```

## Used Tools, Libraries, and Packages

-   **Programming Language:** Python
-   **Core Libraries:**
    -   **Streamlit:** For creating the web-based user interface.
    -   **LangChain:** The primary framework for building the RAG pipeline, including chains, memory, and prompts.
    -   **PyMuPDF (fitz):** For extracting images from PDF files.
    -   **Pillow:** For image manipulation.
    -   **Pytesseract:** For performing OCR on images to extract text.
    -   **FAISS (Facebook AI Similarity Search):** For creating and managing the vector store for efficient similarity search.
    -   **Hugging Face Transformers:** For accessing and using pre-trained embedding models.
    -   **Google Generative AI:** For using the Gemini LLM.
    -   **Django & Django Rest Framework:** For building the backend API.
-   **Models:**
    -   **Embedding Model:** `paraphrase-multilingual-MiniLM-L12-v2` (from Hugging Face)
    -   **LLM:** `gemini-2.5-flash` (from Google)

## Sample Queries and Outputs

Here are some examples:

-   **Query:** "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
-   **Output:** "অনুপমের মামাকে"

## API Documentation

The core logic of the RAG system is organized into several modules in the `rag_code` directory:

-   `extractor.py`: Handles the extraction of text from PDF files. It converts PDF pages to images and then uses Tesseract OCR to extract the text. It also includes extensive text cleaning functions.
-   `chunker.py`: Implements a custom chunking strategy for Bengali and mixed-language text. It splits the text into smaller, semantically meaningful chunks.
-   `embedder.py`: Generates vector embeddings for the text chunks using a pre-trained Hugging Face model.
-   `db.py`: Manages the FAISS vector store, including creating, saving, and loading the vector database.
-   `llm.py`: Contains the main RAG pipeline, including the LLM, conversational memory, and the retrieval chain.
-   `app.py`: The main Streamlit application that provides a user interface for interacting with the RAG system.

## Evaluation Matrix

| Metric                  | Score | Notes                                                                                                                                 |
| ----------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Retrieval Precision** | 00   | Measures the percentage of retrieved documents that are relevant to the query.                                                        |
| **Retrieval Recall**    | 00   | Measures the percentage of all relevant documents that are successfully retrieved.                                                    |
| **Answer Relevance**    | 00   | Assesses how well the generated answer addresses the user's question.                                                                 |
| **Answer Faithfulness** | 00   | Measures whether the generated answer is factually consistent with the information in the retrieved documents.                        |
| **Latency**             | 00   | The time it takes for the system to generate a response after receiving a query.                                                      |
| **Language Support**    | Good  | The system is designed to handle both Bengali and English, with a focus on Bengali-specific text processing and cleaning.             |

---

## Technical Deep Dive

### 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

I used a combination of **PyMuPDF (fitz)** and **Pytesseract** for text extraction. Here's the process:

1.  **PDF to Images:** The `pdf_to_images` function in `extractor.py` uses `fitz.open()` to open the PDF file. It then iterates through the pages, converts each page into a high-resolution image (300 DPI), and stores it as a Pillow `Image` object. This approach was chosen because many Bengali PDFs are image-based or have complex layouts that make direct text extraction unreliable.

2.  **OCR on Images:** The `ocr_images_to_text` function then uses `pytesseract.image_to_string` to perform Optical Character Recognition (OCR) on each image. The language is explicitly set to Bengali (`-l ben`) to improve accuracy.

**Formatting Challenges:**

Yes, I faced several formatting challenges, Especially the font breaking problem:

-   **Complex Layouts:** The PDF contained multi-column layouts, tables, and decorative elements that interfered with standard text extraction. The image-based OCR approach helped mitigate this by treating the page as a single image.
-   **Embedded Fonts:** Some PDFs use non-standard or embedded fonts for Bengali characters, which are not always correctly interpreted by direct text extraction libraries. OCR provides a more robust solution in these cases.
-   **Noise and Artifacts:** Scanned documents often have noise, skew, and other artifacts. While the current implementation doesn't include advanced image pre-processing (like deskewing or noise reduction), the high DPI helps improve the quality of the OCR input.
-   **Incorrect Character Recognition:** Tesseract, while powerful, is not perfect. I had to implement extensive post-processing and cleaning functions (`clean_extracted_text` in `extractor.py`) to correct common OCR errors, remove irrelevant text (like page numbers and headers), and standardize punctuation.

### 2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

I implemented a custom, hybrid chunking strategy in the `BengaliChunker` class. It's primarily **sentence-based** but incorporates several enhancements to improve semantic coherence:

-   **Sentence Splitting:** It starts by splitting the text into sentences using a comprehensive list of Bengali and English punctuation (`।`, `.` , `?`, `!`).
-   **Clause Splitting:** For very long sentences, it attempts to split them into smaller clauses using separators like `,`, `;`, and `|`.
-   **Discourse Marker Splitting:** It also uses a list of Bengali discourse markers (e.g., `কিন্তু`, `তবে`, `সেজন্য`) to identify natural semantic boundaries within the text.
-   **Character Limits:** It uses `max_chunk_size` (400 characters) and `min_chunk_size` (200 characters) to ensure that the chunks are of a reasonable size for the embedding model.
-   **Word-Boundary Aware Overlap:** It creates an overlap of 50 characters between chunks to maintain context. The overlap is created at a word boundary to avoid splitting words in the middle.

**Why it works well for semantic retrieval:**

-   **Preserves Semantic Units:** By splitting on sentences and clauses, the chunker keeps related ideas together. This is crucial for the embedding model to capture the full meaning of the text.
-   **Contextual Overlap:** The overlap ensures that the context is not lost at the boundaries of the chunks, which helps the model understand the relationship between adjacent chunks.
-   **Language-Specific:** The use of Bengali punctuation and discourse markers makes the chunking strategy more effective for Bengali text compared to a generic, language-agnostic approach.
-   **Handles Mixed Languages:** The chunker is designed to handle both Bengali and English text, making it suitable for documents that contain a mix of both languages.

### 3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

I used the **`paraphrase-multilingual-MiniLM-L12-v2`** model from the `sentence-transformers` library.

**Why I chose it:**

-   **Multilingual:** This model is specifically trained on a large corpus of text in over 50 languages, including Bengali and English. This makes it ideal for this project, as it can effectively generate embeddings for both languages.
-   **Performance:** It offers a good balance between performance and size. It's small enough to run efficiently on a CPU, but powerful enough to generate high-quality embeddings.
-   **Semantic Understanding:** It's a "paraphrase" model, which means it's trained to understand the semantic similarity between sentences. It can recognize that two sentences have a similar meaning even if they use different words.
-   **Normalization:** The embeddings are normalized, which makes them suitable for use with similarity metrics like cosine similarity.

**How it captures the meaning of the text:**

The model uses a transformer architecture to process the input text. It converts each chunk of text into a high-dimensional vector (in this case, 384 dimensions). The position of this vector in the vector space represents the semantic meaning of the text. Chunks with similar meanings will have vectors that are close to each other in the vector space.

### 4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

I am using **FAISS (Facebook AI Similarity Search)** as the vector store and **cosine similarity** as the similarity metric.

**Comparison Process:**

1.  When a user asks a question, the query is first converted into an embedding using the same `paraphrase-multilingual-MiniLM-L12-v2` model.
2.  This query embedding is then compared against all the chunk embeddings stored in the FAISS index.
3.  FAISS efficiently finds the `k` most similar chunks (in this case, `k=8`) based on cosine similarity.

**Why this method and storage setup:**

-   **FAISS for Speed:** FAISS is highly optimized for fast similarity search in large-scale vector datasets. It uses indexing techniques to avoid a brute-force search, which would be too slow for a real-time application.
-   **Cosine Similarity for Semantic Search:** Cosine similarity is a good choice for comparing embeddings because it measures the cosine of the angle between two vectors. This is a measure of orientation, not magnitude, which is what we want for semantic similarity. It's effective at finding documents that are semantically related to the query, even if they don't share the exact same keywords.
-   **Local Storage:** The FAISS index is saved locally, which makes it easy to set up and use without the need for a separate database server.

### 5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

**Ensuring Meaningful Comparison:**

-   **Same Embedding Model:** The most important factor is that both the query and the document chunks are embedded using the *same* model. This ensures that they are represented in the same vector space, making the comparison meaningful.
-   **Bilingual Prompt:** The prompt template in `llm.py` is designed to handle both Bengali and English. It explicitly tells the LLM to use the provided context and conversation history to answer the question in the same language as the query.
-   **Conversational Memory:** The `ConversationBufferWindowMemory` keeps track of the last three turns of the conversation. This helps the system understand the context of the query and provide more relevant answers.

**Handling Vague or Missing Context:**

If the query is vague or missing context, the system's performance will depend on a few factors:

-   **If the answer is in the chat history:** The conversational memory might provide enough context for the LLM to understand the query and generate a relevant answer. For example, if the user asks "What about the second one?", the system can look at the previous conversation to understand what "the second one" refers to.
-   **If the answer is not in the chat history:** The system will still try to find the most relevant chunks from the document based on the query's embedding. However, if the query is too vague (e.g., "Tell me more"), the retrieved chunks might not be very helpful. In this case, the LLM will likely generate a more generic answer or ask for clarification. The prompt is designed to be concise and not to explain the answer, so it will likely provide a short, unhelpful answer if the query is too vague.

### 6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?

The relevance of the results is generally good, but there is always room for improvement. Here are some factors that could affect the relevance and how to improve them:

**Potential Issues and Improvements:**

-   **Chunking:**
    -   **Issue:** The current chunking strategy might sometimes split a semantic unit across two chunks, even with the overlap.
    -   **Improvement:** A more advanced, model-based chunking strategy could be used. For example, a model could be trained to identify the optimal split points in the text based on semantic coherence.

-   **Embedding Model:**
    -   **Issue:** While `paraphrase-multilingual-MiniLM-L12-v2` is good, it might not be the best model for all types of content.
    -   **Improvement:** A larger, more powerful embedding model could be used, such as one of the larger models from the `sentence-transformers` library or a commercial model like Cohere's multilingual model. This would likely improve the quality of the embeddings and the relevance of the retrieved chunks. Google's
    new embadding model also can be tested for better results.

-   **Document Size and Quality:**
    -   **Issue:** The quality of the results is highly dependent on the quality and comprehensiveness of the source document. If the document is poorly written, contains errors, or is not relevant to the user's query, the results will be poor.
    -   **Improvement:** Using a larger, more comprehensive, and well-structured document would improve the results.

-   **Retrieval Strategy:**
    -   **Issue:** The current retriever uses a fixed `k` value of 8. This might not be optimal for all queries.
    -   **Improvement:** A more dynamic retrieval strategy could be used. For example, the number of retrieved chunks could be adjusted based on the complexity of the query or the diversity of the retrieved results.

-   **LLM:**
    -   **Issue:** `gemini-2.5-flash` is a fast and capable model, but a more powerful model like `gemini-2.5-pro` might be better at synthesizing information from the retrieved chunks and generating more nuanced answers.
    -   **Improvement:** Experimenting with different LLMs could lead to better results.

By systematically evaluating and improving each of these components, the relevance and overall quality of the RAG system can be significantly enhanced.
