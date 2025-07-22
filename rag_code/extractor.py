import pdfplumber
import pymupdf  # fitz
import re
import unicodedata
from typing import List, Tuple, Optional

def extract_text_multiple_methods(pdf_path: str, start_page: int = 3, end_page: int = 19) -> List[Tuple[str, int, str]]:
    """
    Extract Bengali text using multiple methods to prevent font breaking.
    Returns list of (text, page_number, method_used) tuples.
    """
    results = []
    
    # Method 1: pdfplumber (your current method)
    try:
        plumber_results = extract_with_pdfplumber(pdf_path, start_page, end_page)
        for text, page_num in plumber_results:
            results.append((text, page_num, "pdfplumber"))
    except Exception as e:
        print(f"pdfplumber failed: {e}")
    
    # Method 2: PyMuPDF with font preservation
    try:
        pymupdf_results = extract_with_pymupdf(pdf_path, start_page, end_page)
        for text, page_num in pymupdf_results:
            results.append((text, page_num, "pymupdf"))
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
    
    # Method 3: pdfplumber with character-level extraction
    try:
        char_level_results = extract_with_character_level(pdf_path, start_page, end_page)
        for text, page_num in char_level_results:
            results.append((text, page_num, "character_level"))
    except Exception as e:
        print(f"Character level extraction failed: {e}")
    
    return results

def extract_with_pdfplumber(pdf_path: str, start_page: int, end_page: int) -> List[Tuple[str, int]]:
    """
    Enhanced pdfplumber extraction with better Bengali handling.
    """
    clean_pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            if start_page <= page_number <= end_page:
                # Try different extraction methods
                text = None
                
                # Method 1: Standard extraction
                try:
                    text = page.extract_text()
                except:
                    pass
                
                # Method 2: Extract with layout preservation
                if not text or is_broken_bengali(text):
                    try:
                        text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3)
                    except:
                        pass
                
                # Method 3: Character-by-character extraction
                if not text or is_broken_bengali(text):
                    try:
                        chars = page.chars
                        text = reconstruct_text_from_chars(chars)
                    except:
                        pass
                
                if text:
                    cleaned = clean_bangla_text_advanced(text)
                    if cleaned:
                        clean_pages.append((cleaned, page_number))
    
    return clean_pages

def extract_with_pymupdf(pdf_path: str, start_page: int, end_page: int) -> List[Tuple[str, int]]:
    """
    Extract using PyMuPDF which often handles fonts better.
    """
    clean_pages = []
    
    try:
        doc = pymupdf.open(pdf_path)
        
        for page_num in range(start_page - 1, min(end_page, len(doc))):
            page = doc[page_num]
            
            # Method 1: Standard text extraction
            text = page.get_text()
            
            # Method 2: Extract with font information preserved
            if not text or is_broken_bengali(text):
                text_dict = page.get_text("dict")
                text = reconstruct_from_text_dict(text_dict)
            
            # Method 3: Extract as HTML then parse
            if not text or is_broken_bengali(text):
                html_text = page.get_text("html")
                text = extract_text_from_html(html_text)
            
            if text:
                cleaned = clean_bangla_text_advanced(text)
                if cleaned:
                    clean_pages.append((cleaned, page_num + 1))
        
        doc.close()
    except Exception as e:
        print(f"PyMuPDF extraction error: {e}")
    
    return clean_pages

def extract_with_character_level(pdf_path: str, start_page: int, end_page: int) -> List[Tuple[str, int]]:
    """
    Character-level extraction using pdfplumber for maximum font preservation.
    """
    clean_pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            if start_page <= page_number <= end_page:
                try:
                    chars = page.chars
                    if chars:
                        text = reconstruct_text_from_chars(chars)
                        if text:
                            cleaned = clean_bangla_text_advanced(text)
                            if cleaned:
                                clean_pages.append((cleaned, page_number))
                except Exception as e:
                    print(f"Character level extraction failed for page {page_number}: {e}")
    
    return clean_pages

def reconstruct_text_from_chars(chars: List[dict]) -> str:
    """
    Reconstruct text from character objects, preserving Bengali font integrity.
    """
    if not chars:
        return ""
    
    # Sort characters by position (top to bottom, left to right)
    sorted_chars = sorted(chars, key=lambda x: (round(x['top'], 1), round(x['x0'], 1)))
    
    lines = []
    current_line = []
    current_top = None
    
    for char in sorted_chars:
        char_top = round(char['top'], 1)
        
        # Check if we're on a new line
        if current_top is None:
            current_top = char_top
        elif abs(char_top - current_top) > 3:  # New line threshold
            if current_line:
                lines.append(''.join(current_line))
                current_line = []
            current_top = char_top
        
        # Add character to current line
        text = char.get('text', '')
        if text and text != ' ':  # Skip empty or space-only chars in some contexts
            current_line.append(text)
        elif text == ' ' and current_line:  # Add spaces between words
            current_line.append(' ')
    
    # Add the last line
    if current_line:
        lines.append(''.join(current_line))
    
    return '\n'.join(lines)

def reconstruct_from_text_dict(text_dict: dict) -> str:
    """
    Reconstruct text from PyMuPDF text dictionary.
    """
    text_parts = []
    
    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # Text block
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if span_text:
                        line_text += span_text
                if line_text.strip():
                    text_parts.append(line_text)
    
    return '\n'.join(text_parts)

def extract_text_from_html(html_text: str) -> str:
    """
    Extract clean text from HTML representation.
    """
    import re
    
    # Remove HTML tags but preserve text
    text = re.sub(r'<[^>]+>', '', html_text)
    # Decode HTML entities
    import html
    text = html.unescape(text)
    
    return text

def is_broken_bengali(text: str) -> bool:
    """
    Check if Bengali text appears to be broken (missing conjuncts, improper rendering).
    """
    if not text:
        return True
    
    # Count Bengali characters
    bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    total_chars = len(re.sub(r'\s+', '', text))
    
    if total_chars == 0:
        return True
    
    bengali_ratio = bengali_chars / total_chars
    
    # If there should be Bengali but ratio is too low, might be broken
    if bengali_ratio < 0.1 and any(ord(c) > 127 for c in text):
        return True
    
    # Check for common broken patterns
    broken_patterns = [
        r'[\x00-\x1F]',  # Control characters
        r'[^\u0000-\u007F\u0980-\u09FF\s\.,!?;:()।]',  # Unexpected characters
    ]
    
    for pattern in broken_patterns:
        if re.search(pattern, text):
            return True
    
    return False

def clean_bangla_text_advanced(text: str) -> str:
    """
    Advanced cleaning for Bengali text with font preservation.
    """
    if not text:
        return ""
    
    # Normalize Unicode (important for Bengali)
    text = unicodedata.normalize('NFC', text)
    
    # Remove common PDF artifacts
    text = re.sub(r"পৃষ্ঠা\s*\d+", "", text)               # page numbers
    text = re.sub(r"মডেল\s*টেস্ট.*", "", text)            # repeated headers
    text = re.sub(r"(প্রশ্ন\s*\d+|খণ্ড\s*\d+)", "", text)  # question/section markers
    
    # Fix common spacing issues while preserving Bengali conjuncts
    text = re.sub(r"([^\s])\s+([।!?])", r"\1\2", text)     # Fix spacing before Bengali punctuation
    text = re.sub(r"([।!?])\s*([^\s])", r"\1 \2", text)    # Fix spacing after Bengali punctuation
    
    # Clean excessive whitespace
    text = re.sub(r"[^\S\r\n]+", " ", text)                # multiple spaces to single
    text = re.sub(r"\n{3,}", "\n\n", text)                 # max 2 consecutive newlines
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)  # strip line spaces
    
    # Remove very short lines that are likely artifacts
    lines = [line for line in text.split("\n") if len(line.strip()) > 5]
    cleaned_text = "\n".join(lines)
    
    return cleaned_text.strip()

def compare_extraction_methods(pdf_path: str, start_page: int = 3, end_page: int = 19) -> dict:
    """
    Compare different extraction methods and return the best result.
    """
    all_results = extract_text_multiple_methods(pdf_path, start_page, end_page)
    
    if not all_results:
        return {"best_text": "", "method": "none", "confidence": 0}
    
    # Group results by page
    page_results = {}
    for text, page_num, method in all_results:
        if page_num not in page_results:
            page_results[page_num] = []
        page_results[page_num].append((text, method))
    
    # Choose best result for each page
    best_pages = []
    for page_num in sorted(page_results.keys()):
        page_texts = page_results[page_num]
        
        # Score each text based on quality metrics
        scored_texts = []
        for text, method in page_texts:
            score = score_bengali_text_quality(text)
            scored_texts.append((text, method, score))
        
        # Choose the best scoring text
        best_text, best_method, best_score = max(scored_texts, key=lambda x: x[2])
        best_pages.append((best_text, page_num, best_method))
    
    # Combine all pages
    combined_text = "\n\n".join([text for text, _, _ in best_pages])
    methods_used = [method for _, _, method in best_pages]
    most_common_method = max(set(methods_used), key=methods_used.count)
    
    return {
        "best_text": combined_text,
        "method": most_common_method,
        "page_details": best_pages,
        "confidence": sum(score_bengali_text_quality(text) for text, _, _ in best_pages) / len(best_pages)
    }

def score_bengali_text_quality(text: str) -> float:
    """
    Score the quality of extracted Bengali text.
    """
    if not text:
        return 0.0
    
    score = 0.0
    
    # Bengali character ratio
    bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    total_chars = len(re.sub(r'\s+', '', text))
    if total_chars > 0:
        bengali_ratio = bengali_chars / total_chars
        score += bengali_ratio * 40  # Up to 40 points
    
    # Text length (longer is often better)
    score += min(len(text) / 100, 20)  # Up to 20 points
    
    # Proper sentence structure (ending with Bengali punctuation)
    sentences_with_punctuation = len(re.findall(r'[।!?]', text))
    total_sentences = len(re.findall(r'[।!?।\.\?!]', text))
    if total_sentences > 0:
        punctuation_ratio = sentences_with_punctuation / total_sentences
        score += punctuation_ratio * 20  # Up to 20 points
    
    # Penalty for broken characters or artifacts
    if re.search(r'[\x00-\x1F]', text):  # Control characters
        score -= 10
    
    if re.search(r'[^\u0000-\u007F\u0980-\u09FF\s\.,!?;:()।\-]', text):  # Unexpected chars
        score -= 5
    
    # Penalty for too many single characters (likely broken)
    single_char_lines = len([line for line in text.split('\n') if len(line.strip()) == 1])
    total_lines = len([line for line in text.split('\n') if line.strip()])
    if total_lines > 0 and single_char_lines / total_lines > 0.3:
        score -= 15
    
    return max(0.0, score)

# Usage example
if __name__ == "__main__":
    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    
    # Get the best extraction result
    result = compare_extraction_methods(pdf_path, start_page=3, end_page=19)
    
    print(f"Best method: {result['method']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Extracted text length: {len(result['best_text'])}")
    print("\nFirst 500 characters:")
    print(result['best_text'][:500])
    
    # Save to file
    with open('cleaned_text.txt', 'w', encoding='utf-8') as f:
        f.write(result['best_text'])