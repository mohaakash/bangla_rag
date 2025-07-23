import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import io
import os

# Configure Tesseract path (Windows) - update this to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# OCR configuration for Bengali
TESSERACT_CONFIG = '--psm 6 -l ben'  # Bengali language, page segmentation mode 6

def pdf_to_images(pdf_path, dpi=300, start_page=None, end_page=None):
    """
    Convert PDF pages to images using PyMuPDF (fitz)
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image conversion (higher = better quality but slower)
        start_page: Starting page number (1-based index)
        end_page: Ending page number (1-based index)
    
    Returns:
        List of PIL Image objects
    """
    images = []
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # Convert to 0-based index and handle None values
        start = (start_page - 1) if start_page is not None else 0
        end = end_page if end_page is not None else len(doc)
        
        # Validate page range
        if start < 0 or end > len(doc) or start >= end:
            raise ValueError(f"Invalid page range. PDF has {len(doc)} pages.")
        
        for page_num in range(start, end):
            # Get the page
            page = doc.load_page(page_num)
            
            # Create a matrix for conversion (controls DPI/quality)
            mat = fitz.Matrix(dpi/72, dpi/72)  # 72 is the default PDF DPI
            
            # Get the page as a pixel map
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return None

def ocr_images_to_text(images):
    """
    Perform OCR on a list of images
    
    Args:
        images: List of PIL Image objects
    
    Returns:
        List of extracted texts (one per page)
    """
    extracted_text = []
    
    for i, image in enumerate(images):
        try:
            # Convert image to grayscale for better OCR
            gray_image = image.convert('L')
            
            # Perform OCR
            text = pytesseract.image_to_string(gray_image, config=TESSERACT_CONFIG)
            extracted_text.append(text)
            print(f"Processed page {i+1}/{len(images)}")
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
            extracted_text.append("")  # Add empty string if OCR fails
    
    return extracted_text

def extract_text_from_pdf(pdf_path: str, start_page: int = None, end_page: int = None) -> str:
    """
    Extracts text from specified pages of a PDF file by converting them to images and using OCR.

    Args:
        pdf_path: The path to the PDF file.
        start_page: Starting page number (1-based index)
        end_page: Ending page number (1-based index)

    Returns:
        A single string containing all the extracted text.
    """
    print(f"Starting PDF text extraction process for pages {start_page or 1} to {end_page or 'end'}...")
    images = pdf_to_images(pdf_path, start_page=start_page, end_page=end_page)
    if not images:
        print("Failed to convert PDF to images. Exiting.")
        return ""
    
    print("Performing OCR on images...")
    text_list = ocr_images_to_text(images)
    
    full_text = "\n".join(text_list)
    print("Text extraction completed.")
    return full_text

