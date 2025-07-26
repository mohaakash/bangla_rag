import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    faithfulness,
    context_recall,
    context_precision
)
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import required Google AI components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Initialize Gemini LLM and Embeddings
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    max_tokens=None,
    top_p=0.8
)

gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Wrap for RAGAS
ragas_llm = LangchainLLMWrapper(gemini_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

# Load your data with corrected column names
data = {
    "question": [  # Changed from "Question"
        "মঙ্গলঘট ভরা ছিল' উক্তি দ্বারা কী ইঙ্গিত করা হয়েছে?",
        "রবীন্দ্রনাথ ঠাকুর কত খষ্টাব্দে মৃত্যুবরণ করেন?",
        "অনুপমের বয়স কত?",
        "অনুপমের মতে, তার প্রকৃত ভাগ্যদেবতা কে?",
        "অপরিচিতা' গল্পে কোন সামাজিক প্রথার সমালোচনা করা হয়েছে?",
        "অনুপমের বাবা কী কাজ করে জীবিকা নির্বাহ করতেন?",
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ],
    "answer": [  # Changed from "LLm_answer"
        "শুভ ও পূর্ণতার ইঙ্গিত করা হয়েছে।",
        "১৯৪১ খষ্টাব্দে",
        "অনুপমের বয়স সাতাশ বছর",
        "মামা।",
        "যৌতুক প্রথা।",
        "অনুপমের বাবা ওকালতি করে জীবিকা নির্বাহ করতেন।",
        "প্রদত্ত তথ্যে নেই।",
        "অনুপমের মামাকে",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স পনেরো বছর ছিল"
    ],
    "ground_truth": [  # Changed from "Ground_Truth"
        "মঙ্গলঘট ভরা ছিল' উক্তি দ্বারা কল্যাণীর বংশে একসময় লক্ষ্মীর কৃপা ও ঐশ্বর্যপূর্ণ থাকার ইঙ্গিত করা হয়েছে।",
        "১৯৪১",
        "সাতাশ বছর",
        "মামা।",
        "যৌতুক প্রথা।",
        "ওকালতি",
        "শুম্ভুনাথ",
        "মামাকে",
        "১৫ বছর"
    ],
    "contexts": [
        ["ব্যাংকের লক্ষ্মীর একসময় লক্ষ্মীর কৃপা ঐশ্বর্যের ঘট পূর্ণ ছিল। মঙ্গলঘট ভরা ছিল।"],
        ["রবীন্দ্রনাথ ঠাকুর। জন্ম তারিখ: ৭ মে, ১৮৬১ খ্রিষ্টাব্দ (২৫ বৈশাখ, ॥ জন্ম পরিচয় | ১২৬৮ বঙ্গাব্দ)"],
        ["আর আমার বয়স সাতাশ মাত্র।"],
        ["অনেক বড়ো ঘর হইতে আমার সম্বন্ধ আসিয়াছিল। কিন্তু মামা, যিনি পৃথিবীতে আমার ভাগ্যদেবতার প্রধান এজেন্ট"],
        ["যৌতুক নিতে বারবার বলায় কন্যার পিতা অপমানিত বোধ করে।"],
        ["আমার পিতা এককালে ব্যারিস্টার ছিলেন। ওকালতি করে তিনি প্রচুর টাকা রোজগার করেছিলেন।"],
        ["কন্যার পিতা শস্তুনাথবাবু হরিশকে কত বিশ্বাস করেন তাহার প্রমাণ এই যে, বিবাহের তিন দিন পূর্বে তিনি আমাকে চক্ষে দেখেন এবং আশীর্বাদ করিয়া যান। বয়স তার চল্লিশের কিছু এপারে বা ওপারে। চুল কীচা, গোঁফে পাক ধরিতে আরম্ভ করিয়াছে মাত্র। সুপুরুষ বটে। ভিড়ের মধ্যে দেখিলে সকলের আগে তার উপরে চোখ পড়িবার মতো চেহারা। আশা করি আমাকে দেখিয়া তিনি খুশি হইয়াছিলেন। বোঝা শক্ত, কেননা তিনি বড়ই চুপচাপ।"],
        ["মামাকে ভাগ্য দেবতার প্রধান এজেন্ট বলার কারণ, তার- উত্তর: প্রভাব । কিন্তু মামা, যিনি পৃথিবীতে আমার ভাগ্যদেবতার প্রধান এজেন্ট, বিবাহ সম্বন্ধে তার একটা বিশেষ মত ছিল।"],
        ["এসব ভালো কথা। কিন্তু, মেয়ের বয়স যে পনেরো, তাই শুনিয়া মামার মন ভার হইল।"]
    ]
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Define metrics
metrics = [
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    faithfulness,
    context_recall,
    context_precision
]

try:
    # Evaluate with Gemini
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    
    # Convert to dataframe and save results
    results_df = result.to_pandas()
    results_df.to_csv('ragas_evaluation_results.csv', index=False)
    
    # Print overall scores
    print("\nOverall Evaluation Scores:")
    print(f"Answer Relevancy: {result['answer_relevancy']:.2f}")
    print(f"Answer Correctness: {result['answer_correctness']:.2f}")
    print(f"Answer Similarity: {result['answer_similarity']:.2f}")
    print(f"Faithfulness: {result['faithfulness']:.2f}")
    print(f"Context Recall: {result['context_recall']:.2f}")
    print(f"Context Precision: {result['context_precision']:.2f}")
    
    # Print detailed results
    print("\nDetailed Results:")
    print(results_df.to_string())
    
except Exception as e:
    print(f"Error during evaluation: {e}")
    print("Make sure you have configured your LLM API keys properly.")