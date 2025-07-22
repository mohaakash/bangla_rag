import sys
import codecs
import os
import shutil

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from db import create_chroma_collection, insert_text_to_chroma
from rag import setup_gemini, answer_question

# Define the path to the cleaned text file
CLEANED_TEXT_FILE = os.path.join(os.path.dirname(__file__), "..", "cleaned_text.txt")

# Define the Chroma store directory
CHROMA_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")

# Delete the existing Chroma collection to ensure fresh data with new chunking
if os.path.exists(CHROMA_STORE_DIR):
    shutil.rmtree(CHROMA_STORE_DIR)
    print(f"Deleted existing Chroma store at {CHROMA_STORE_DIR}")

collection = create_chroma_collection()
model = setup_gemini()

# Read the cleaned text
with open(CLEANED_TEXT_FILE, "r", encoding="utf-8") as f:
    cleaned_text = f.read()

# Insert the cleaned text into ChromaDB using the new chunking strategy
# Assuming the text is primarily Bengali, set language='bengali'
insert_text_to_chroma(cleaned_text, collection, language="bengali")

question = "কাকে অনুপমের ভাগ্যদেবতা বলে উল্লেখ করা হয়েছে?"
response = answer_question(collection, question, model)

print("Gemini Answer:", response)
