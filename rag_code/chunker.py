import nltk
import re
from nltk.tokenize import sent_tokenize
from typing import List

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for Gemini models.
    Gemini typically uses ~4 characters per token for English, ~2-3 for Bengali.
    This is an approximation since we don't have access to Gemini's exact tokenizer.
    """
    # Count Bengali characters (Unicode range for Bengali)
    bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    # Count other characters
    other_chars = len(text) - bengali_chars
    
    # Rough estimation: Bengali ~2.5 chars/token, English/other ~4 chars/token
    estimated_tokens = (bengali_chars / 2.5) + (other_chars / 4)
    return int(estimated_tokens)

def improved_bengali_sentence_split(text: str) -> List[str]:
    """
    Improved Bengali sentence splitting that handles more cases.
    """
    # Bengali sentence enders: দাঁড়ি (।), প্রশ্ন চিহ্ন (?), বিস্ময় চিহ্ন (!)
    # Also include English punctuation for mixed content
    pattern = r'[।!?\.]+(?:\s|$)'
    
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def detect_language_mix(text: str) -> str:
    """
    Detect if text is primarily Bengali, English, or mixed.
    """
    bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    total_chars = len(re.sub(r'\s+', '', text))
    
    if total_chars == 0:
        return 'english'
    
    bengali_ratio = bengali_chars / total_chars
    
    if bengali_ratio > 0.7:
        return 'bengali'
    elif bengali_ratio > 0.1:
        return 'mixed'
    else:
        return 'english'

def chunk_text_for_gemini(
    text: str, 
    max_tokens: int = 1000,  # Gemini 2.5 Flash can handle larger chunks
    overlap_tokens: int = 100,
    min_chunk_size: int = 50
) -> List[str]:
    """
    Chunk text optimized for Gemini 2.5 Flash with Bengali and English support.
    
    Args:
        text (str): Input text to chunk
        max_tokens (int): Maximum tokens per chunk (default 1000 for Gemini)
        overlap_tokens (int): Overlap between chunks
        min_chunk_size (int): Minimum tokens for a chunk
    
    Returns:
        List[str]: List of text chunks
    """
    if not text.strip():
        return []
    
    # Detect language for appropriate processing
    language = detect_language_mix(text)
    
    # Split into sentences based on detected language
    if language == 'bengali':
        sentences = improved_bengali_sentence_split(text)
    elif language == 'mixed':
        # For mixed content, try NLTK first, then Bengali splitting
        try:
            sentences = sent_tokenize(text, language='english')
            # Further split any sentences that might contain Bengali punctuation
            refined_sentences = []
            for sent in sentences:
                if re.search(r'[।]', sent):
                    refined_sentences.extend(improved_bengali_sentence_split(sent))
                else:
                    refined_sentences.append(sent)
            sentences = refined_sentences
        except:
            sentences = improved_bengali_sentence_split(text)
    else:  # English
        sentences = sent_tokenize(text, language='english')
    
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_tokens = estimate_tokens(sentence)
        
        # If single sentence exceeds max_tokens, split it further
        if sentence_tokens > max_tokens:
            # Split by clauses or phrases
            parts = re.split(r'[,;:|]', sentence)
            for part in parts:
                part = part.strip()
                if part:
                    part_tokens = estimate_tokens(part)
                    if current_tokens + part_tokens > max_tokens and current_chunk:
                        # Finalize current chunk
                        chunks.append(' '.join(current_chunk))
                        
                        # Handle overlap
                        if overlap_tokens > 0 and chunks:
                            overlap_text = []
                            overlap_token_count = 0
                            
                            # Add sentences from end of current chunk for overlap
                            for i in range(len(current_chunk) - 1, -1, -1):
                                sent_tokens = estimate_tokens(current_chunk[i])
                                if overlap_token_count + sent_tokens <= overlap_tokens:
                                    overlap_text.insert(0, current_chunk[i])
                                    overlap_token_count += sent_tokens
                                else:
                                    break
                            
                            current_chunk = overlap_text[:]
                            current_tokens = overlap_token_count
                        else:
                            current_chunk = []
                            current_tokens = 0
                    
                    current_chunk.append(part)
                    current_tokens += part_tokens
            continue
        
        # Check if adding this sentence would exceed the limit
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Finalize current chunk
            chunks.append(' '.join(current_chunk))
            
            # Handle overlap
            if overlap_tokens > 0:
                overlap_text = []
                overlap_token_count = 0
                
                # Add sentences from end of current chunk for overlap
                for i in range(len(current_chunk) - 1, -1, -1):
                    sent_tokens = estimate_tokens(current_chunk[i])
                    if overlap_token_count + sent_tokens <= overlap_tokens:
                        overlap_text.insert(0, current_chunk[i])
                        overlap_token_count += sent_tokens
                    else:
                        break
                
                current_chunk = overlap_text[:]
                current_tokens = overlap_token_count
            else:
                current_chunk = []
                current_tokens = 0
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add the final chunk
    if current_chunk:
        final_chunk = ' '.join(current_chunk)
        if estimate_tokens(final_chunk) >= min_chunk_size:
            chunks.append(final_chunk)
        elif chunks:  # Merge with previous chunk if too small
            chunks[-1] += ' ' + final_chunk
        else:
            chunks.append(final_chunk)
    
    return chunks

def analyze_chunks(chunks: List[str], text_type: str = ""):
    """
    Analyze and display chunk information.
    """
    print(f"\n--- {text_type} Chunks Analysis ---")
    print(f"Total chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        token_count = estimate_tokens(chunk)
        char_count = len(chunk)
        print(f"Chunk {i}: {token_count} tokens, {char_count} characters")
        print(f"Preview: {chunk[:100]}{'...' if len(chunk) > 100 else ''}\n")

# Example usage
if __name__ == "__main__":
    # Test with Bengali text
    bengali_text = """
    বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ। এটি ভারত ও মিয়ানমার দ্বারা বেষ্টিত। 
    বাংলাদেশের রাজধানী ঢাকা। এখানে অনেক মানুষ বাস করে। 
    বাংলা ভাষা এখানকার প্রধান ভাষা। এটি একটি সুন্দর দেশ।
    এখানে অনেক নদী রয়েছে। পদ্মা, যমুনা, মেঘনা প্রধান নদী।
    """
    
    # Test with English text
    english_text = """
    Bangladesh is a country in South Asia. It is surrounded by India and Myanmar.
    The capital of Bangladesh is Dhaka. Many people live here.
    Bengali is the main language here. It is a beautiful country.
    There are many rivers here. Padma, Jamuna, Meghna are the main rivers.
    """
    
    # Test with mixed content
    mixed_text = """
    Bangladesh বা বাংলাদেশ is a South Asian country। It has a rich cultural heritage.
    The people here speak Bengali বা বাংলা ভাষা। Dhaka ঢাকা is the capital city.
    """
    
    # Test chunking
    bengali_chunks = chunk_text_for_gemini(bengali_text, max_tokens=50, overlap_tokens=10)
    analyze_chunks(bengali_chunks, "Bengali")
    
    english_chunks = chunk_text_for_gemini(english_text, max_tokens=50, overlap_tokens=10)
    analyze_chunks(english_chunks, "English")
    
    mixed_chunks = chunk_text_for_gemini(mixed_text, max_tokens=50, overlap_tokens=10)
    analyze_chunks(mixed_chunks, "Mixed")