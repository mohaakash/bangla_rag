import sys
sys.path.append(r'C:\Users\Mohammad Akash\Documents\projects\bangla_rag')
from rag_code.extractor import extract_text_from_pdf

def export_extracted_text_to_file(pdf_path, txt_output_path, start_page=3, end_page=19):
    """
    Extracts and saves cleaned text from a PDF to a .txt file with UTF-8 encoding.
    """
    pages = extract_text_from_pdf(pdf_path, start_page, end_page)

    with open(txt_output_path, 'w', encoding='utf-8') as f:
        for text, page_num in pages:
            f.write(f"\n\n--- 📄 Page {page_num} ---\n\n")
            f.write(text)
            f.write("\n")

    print(f"Saved extracted Bangla text to: {txt_output_path}")

# 🔧 Run the test
if __name__ == "__main__":
    PDF_PATH = "HSC26-Bangla1st-Paper.pdf"
    OUTPUT_TXT_PATH = "cleaned_text.txt"

    export_extracted_text_to_file(PDF_PATH, OUTPUT_TXT_PATH)
