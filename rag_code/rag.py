import os
from dotenv import load_dotenv
import google.generativeai as genai
from db import query_chroma

# Load the API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def setup_gemini(model_name="models/gemini-2.5-flash"):
    """
    Initializes the Gemini model using the API key from environment.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(model_name)

def build_prompt(context_chunks, user_question):
    context = "\n\n".join(context_chunks)
    prompt = f"""
Answer the following question based on the provided context.First translate the context and question to English if it is in Bengali,
then answer the question in Bengali and English.
If the answer is not in the context, respond with 'তথ্য পাওয়া যায়নি. added with the context.'

Context:
{context}

Question: {user_question}
Answer:"""
    return prompt.strip()


def answer_question(collection, user_query, model, top_k=5):
    """
    Retrieve top_k chunks from Chroma and generates answer using Gemini.
    """
    results = query_chroma(collection, user_query, n_results=top_k)
    if not results["documents"] or not results["documents"][0]:
        return "তথ্য পাওয়া যায়নি।"

    context_chunks = results["documents"][0]
    prompt = build_prompt(context_chunks, user_query)

    response = model.generate_content(prompt)
    return response.text

