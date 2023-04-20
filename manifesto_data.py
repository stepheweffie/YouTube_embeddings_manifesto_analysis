from pdfminer.high_level import extract_text


# Function to extract text from PDF file
def extract_pdf_text(file_path):
    text = extract_text(file_path)
    return text

