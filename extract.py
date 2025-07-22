
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from io import BytesIO
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' 

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)  # Render page as image
        img = Image.open(BytesIO(pix.tobytes("png")))

        # OCR using Bengali language
        text = pytesseract.image_to_string(img, lang="ben")
        full_text += f"{text.strip()}"

    return full_text.strip()

# def extract_text_from_pdf(file_path):
#     text = ""

#     # Step 1: Open PDF using both pdfplumber and PyMuPDF
#     with pdfplumber.open(file_path) as pdf:
#         doc = fitz.open(file_path)

#         for i, page in enumerate(pdf.pages):
#             page_text = page.extract_text() or ""
#             text += f"\n{page_text}\n"

#             # Use PyMuPDF to extract images from the same page
#             page_fitz = doc.load_page(i)
#             images = page_fitz.get_images(full=True)
#             for img_index, img in enumerate(images):
#                 xref = img[0]
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image = Image.open(BytesIO(image_bytes))

#                 # Run OCR on the image
#                 ocr_text = pytesseract.image_to_string(image, lang="ben")
#                 if ocr_text.strip():
#                     text += f"\n{ocr_text.strip()}\n"

#     return text


# What method or library did you use to extract the text, and why?
# A: pdfplumber was chosen because it handles Bengali text encoding more accurately than PyPDF2 and preserves layout better. Some formatting issues were encountered with heading alignment, but overall line breaks and paragraphs were clean.
