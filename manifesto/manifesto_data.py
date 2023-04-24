from pdfminer.high_level import extract_text
import os
import openai
import pandas as pd
from openai_tasks.count_tokens import count_embedding_tokens
from dotenv import load_dotenv
import re
import tiktoken
import spacy
load_dotenv()
nlp = spacy.load('en_core_web_sm')
api_key = os.getenv('OPENAI_API_KEY')
pdf = os.getenv('PDF_FILE_PATH')
pdf_text = os.getenv('PDF_TEXT')

# Function to extract text from PDF file


def extract_pdf_text(file_path):
    text = extract_text(file_path)
    # print(text)
    with open('manifesto.txt', 'w') as f:
        f.write(text)
        return f


def sentence_df():
    extract_pdf_text(pdf)
    with open('manifesto.txt', 'r') as text:
        pdf_sentences = text.read()
        # Tokenize the text into sentences
        sentences = nlp(pdf_sentences)
        # print(sentences)
        # Clean the sentences by removing special characters and numbers
        re.sub(r'\s+', ' ', pdf_sentences).strip()
        doc_sents = [sent.text for sent in sentences.sents]
        # Add the cleaned sentences to the dataframe
        df = pd.DataFrame(data=doc_sents, columns=["sentences"])
        return df


def map_tokens():
    token_map = list()
    text = sentence_df()
    text = text['sentences'].tolist()
    tokens = count_embedding_tokens(text)
    decoder = tiktoken.encoding_for_model(os.getenv("OPENAI_MODEL"))
    for token in tokens:
        token_map.append(decoder.decode(token))
    token_map = [tokens, token_map]
    return token_map


def openai_pdf_embedding():
    # Maximum tokens for the text-embedding-ada-002 model is 8191 per call
    text = map_tokens()
    text = text[1]
    embeddings = list()
    openai.api_key = api_key
    for chunk in text:

        embedding = openai.Embedding.create(
            input=chunk, model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        embeddings += embedding

    df = pd.DataFrame(data=embeddings)
    df['sent_tokens'] = text[0]
    print(df.head())
    return df


openai_pdf_embedding()
# map_tokens()

