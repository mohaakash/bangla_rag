import pytesseract
import pymupdf
from PIL import Image
import io
import os
import re
from datetime import datetime

# Configure Tesseract path (Windows) - update this to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# OCR configuration for Bengali
TESSERACT_CONFIG = '--psm 6 -l ben'  # Bengali language, page segmentation mode 6

#Text Cleaning Functions
def is_mostly_bangla(text, threshold=0.6):
    if not text.strip():
        return False
    bangla_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    total_chars = len([c for c in text if c.isalnum() or '\u0980' <= c <= '\u09FF'])
    return (bangla_chars / max(total_chars, 1)) > threshold

def clean_general_text(text):
    text = re.sub(r'\[([^\]]*(?:বি\.|ইউনিট|বিশ্ববিদ্যালয়|কলেজ|২০\d{2})[^\]]*)\]', '', text)
    text = re.sub(r'\[[^\]]*\d{4}[^\]]*\]', '', text)
    text = re.sub(r'\[([^\]]*(?:রা\.|চা\.|ঢা\.|সি\.|কু\.)[^\]]*)\]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)(?!\d+।)', ' ', text)
    text = re.sub(r'\s+([।,;:])', r'\1', text)
    text = re.sub(r'([।,;:])\s+', r'\1 ', text)
    return text.strip()

def keep_only_correct_mcqs(text):
    mcq_pattern = re.compile(
        r'(\d+।[^(]*?)\s*'
        r'\(?\s*ক\s*\)\s*([^(]*?)\s*'
        r'\(?\s*খ\s*\)\s*([^(]*?)\s*'  
        r'\(?\s*গ\s*\)\s*([^(]*?)\s*'
        r'\(?\s*ঘ\s*\)\s*([^(]*?)\s*'
        r'উত্তর\s*[:：]?\s*\(?([কখগঘ])\)?',
        re.DOTALL | re.IGNORECASE
    )
    
    def choice_to_index(ch):
        return {'ক': 0, 'খ': 1, 'গ': 2, 'ঘ': 3}.get(ch.strip(), -1)
    
    def mcq_replacer(match):
        question = match.group(1).strip()
        options = [match.group(i).strip() for i in range(2, 6)]
        correct_choice = match.group(6).strip()
        correct_idx = choice_to_index(correct_choice)
        
        if 0 <= correct_idx < 4:
            correct_answer = options[correct_idx]
            question = re.sub(r'\s+', ' ', question).strip()
            correct_answer = re.sub(r'\s+', ' ', correct_answer).strip()
            return f"{question}\nউত্তর: {correct_answer}\n"
        else:
            return match.group(0)
    
    return re.sub(mcq_pattern, mcq_replacer, text)

def clean_non_mcq_content(text):
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append('')
            continue
        
        if len(re.findall(r'[0-9]', line)) > len(line) * 0.3:
            continue
        
        if len(line) < 3 and not re.match(r'\d+।', line):
            continue
        
        if is_mostly_bangla(line) or re.match(r'\d+।', line) or line.startswith('উত্তর:'):
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines)
    return re.sub(r'\n{3,}', '\n\n', result)

def final_cleanup(text):
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\d+)\s*।\s*', r'\1। ', text)
    text = re.sub(r'উত্তর\s*[:：]\s*', 'উত্তর: ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()

def clean_extracted_text(text):
    """Main function to clean the extracted text with all steps"""
    print("🔧 Cleaning extracted text...")
    text = clean_general_text(text)
    text = keep_only_correct_mcqs(text)
    text = clean_non_mcq_content(text)
    text = final_cleanup(text)
    return text

#PDF Processing Functions
def pdf_to_images(pdf_path, dpi=300, start_page=None, end_page=None):
    images = []
    try:
        doc = pymupdf.open(pdf_path)
        start = (start_page - 1) if start_page is not None else 0
        end = end_page if end_page is not None else len(doc)
        
        if start < 0 or end > len(doc) or start >= end:
            raise ValueError(f"Invalid page range. PDF has {len(doc)} pages.")
        
        exclude_start = 33
        exclude_end = 41
        
        for page_num in range(start, end):
            if (page_num + 1) >= exclude_start and (page_num + 1) <= exclude_end:
                continue
                
            page = doc.load_page(page_num)
            mat = pymupdf.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return None

def ocr_images_to_text(images):
    extracted_text = []
    for i, image in enumerate(images):
        try:
            gray_image = image.convert('L')
            text = pytesseract.image_to_string(gray_image, config=TESSERACT_CONFIG)
            extracted_text.append(text)
            print(f"Processed page {i+1}/{len(images)}")
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
            extracted_text.append("")
    return extracted_text

def extract_text_from_pdf(pdf_path: str, start_page: int = None, end_page: int = None) -> str:
    print(f"Starting PDF text extraction process for pages {start_page or 1} to {end_page or 'end'}...")
    images = pdf_to_images(pdf_path, start_page=start_page, end_page=end_page)
    if not images:
        print("Failed to convert PDF to images. Exiting.")
        return ""
    
    print("Performing OCR on images...")
    text_list = ocr_images_to_text(images)
    
    full_text = "\n".join(text_list)
    
    # Clean the extracted text before returning
    full_text = clean_extracted_text(full_text)
    
    print("Text extraction and cleaning completed.")
    return full_text