import spacy
from dotenv import load_dotenv
import os
import openai
from manifesto.manifesto_data import extract_pdf_text
load_dotenv()
nlp = spacy.load('en_core_web_sm')



