import pdfplumber
from docx import Document

def read_pdf(file_path):
    """
    Extracts text from a PDF file.
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def read_docx(file_path):
    """
    Extracts text from a Word document (.docx).
    """
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
