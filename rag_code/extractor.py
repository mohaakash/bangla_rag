import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import io
import os

# Configure Tesseract path (Windows) - update this to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# OCR configuration for Bengali
TESSERACT_CONFIG = '--psm 6 -l ben'  # Bengali language, page segmentation mode 6

def pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF pages to images using PyMuPDF (fitz)
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image conversion (higher = better quality but slower)
    
    Returns:
        List of PIL Image objects
    """
    images = []
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
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

def save_text_to_file(text_list, output_file):
    """
    Save extracted text to a file
    
    Args:
        text_list: List of texts (one per page)
        output_file: Path to output text file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, text in enumerate(text_list):
                f.write(f"=== Page {i+1} ===\n")
                f.write(text)
                f.write("\n\n")
        print(f"Successfully saved extracted text to {output_file}")
    except Exception as e:
        print(f"Error saving to file: {e}")

def main():
    # Input and output paths
    pdf_path = "HSC26-Bangla1st-Paper.pdf"  # Change to your PDF file
    output_file = "extracted_text.txt"
    
    # Step 1: Convert PDF to images
    print("Converting PDF to images...")
    images = pdf_to_images(pdf_path, dpi=300)
    if not images:
        print("Failed to convert PDF to images. Exiting.")
        return
    
    # Step 2: Perform OCR on images
    print("Performing OCR on images...")
    extracted_text = ocr_images_to_text(images)
    
    # Step 3: Save extracted text
    print("Saving extracted text...")
    save_text_to_file(extracted_text, output_file)
    
    print("Process completed! Check the output file.")

if __name__ == "__main__":
    main()