import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image
import io
import time

PDF_PATH = "hsc_bangla.pdf"
OUTPUT_PATH = "hsc_bangla_clean.txt"

print("Initializing EasyOCR... This might take a moment to download the models.")
# Initialize the OCR reader for Bengali and English
# The gpu=False flag can sometimes help avoid warnings if you don't have a GPU.
reader = easyocr.Reader(['bn', 'en'], gpu=False)
print("EasyOCR initialized.")

def perform_ocr_on_pdf(pdf_path, output_path):
    """
    Opens a PDF, performs OCR on each page, and saves the extracted text to a file.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    
    print(f"Starting OCR processing for {len(doc)} pages in '{pdf_path}'...")
    start_time = time.time()

    for page_num, page in enumerate(doc):
        print(f"  - Processing page {page_num + 1}/{len(doc)}...")
        
        # Convert page to an image (pixmap)
        pix = page.get_pixmap(dpi=300) # Higher DPI for better OCR accuracy
        
        # Convert pixmap to a format that PIL can handle
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        
        # Convert PIL image to numpy array for EasyOCR
        img_np = np.array(image)
        
        # Perform OCR. 'paragraph=True' helps group text logically.
        result = reader.readtext(img_np, paragraph=True)
        
        # **THE FIX IS HERE**
        # When paragraph=True, result is a list of (bounding_box, text).
        # We only need the 'text' part.
        page_text = "\n".join([text for (_, text) in result])
        
        full_text += page_text + "\n\n" # Add double newline to separate pages

    doc.close()
    
    # Save the final extracted text to the output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    end_time = time.time()
    print("\nOCR processing complete!")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
    print(f"Clean text saved to '{output_path}'.")

if __name__ == "__main__":
    perform_ocr_on_pdf(PDF_PATH, OUTPUT_PATH)