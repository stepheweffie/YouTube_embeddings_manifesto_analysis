import spacy
from dotenv import load_dotenv
import os
import openai
from manifesto.manifesto_data import extract_pdf_text
load_dotenv()
nlp = spacy.load('en_core_web_sm')


def openai_pdf_embedding():
    pdf = os.getenv('PDF_FILE_PATH')
    text = extract_pdf_text(pdf)
    embedding = openai.Embedding.create(
        input=text, model="text-embedding-ada-002"
    )["data"][0]["embedding"]
    return embedding
