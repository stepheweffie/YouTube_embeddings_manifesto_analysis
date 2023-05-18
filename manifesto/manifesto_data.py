from pdfminer.high_level import extract_text
import os
import openai
import pandas as pd
from openai_tasks.count_tokens import count_embedding_tokens
from dotenv import load_dotenv
import re
import tiktoken
import spacy
from visuals.dimensionality_visuals import pca_kmeans_dimensionality
load_dotenv()

nlp = spacy.load('en_core_web_sm')
api_key = os.getenv('OPENAI_API_KEY')
pdf = os.getenv('PDF_FILE_PATH')
pdf_text = os.getenv('PDF_TEXT')

# Function to extract text from PDF file


def extract_pdf_text(file_path):
    text = extract_text(file_path)
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
        re.sub(r'\n+', ' ', pdf_sentences).strip()
        doc_sents = [sent.text for sent in sentences.sents]
        df = pd.DataFrame(data=doc_sents, columns=["sentences"])
        # Remove hyperlinks rows
        mask = df['sentences'].str.contains("hyperlinks")
        # Slice the DataFrame to exclude the rows that contain the word "Hyperlink"
        df = df[~mask]
        print(len(df))
        return df


def map_tokens_df():
    token_map = list()
    text = sentence_df()
    text = text['sentences'].tolist()
    tokens = count_embedding_tokens(text)
    decoder = tiktoken.encoding_for_model(os.getenv("OPENAI_MODEL"))
    for token_list in tokens:
        print(len(token_list))
        token_map.append(decoder.decode(token_list))
    # Build a dataframe
    df = pd.DataFrame()
    df['decoded_tokens'] = token_map
    df['encoded_tokens'] = tokens
    # token_map = [tokens, token_map]
    # print(len(tokens), len(token_map))
    # print(len(df), df[50:100])
    return df


def openai_pdf_embeddings():
    # Maximum tokens for the text-embedding-ada-002 model is 8191 per call
    text = map_tokens_df()
    # Use encoded tokens
    text = text['encoded_tokens']
    embeddings = list()
    openai.api_key = api_key
    for chunk in text[:-1]:
        embedding = openai.Embedding.create(
            input=chunk, model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        embeddings.append(embedding)
        print(len(embedding))
    # A DataFrame with 1536 (embeddings) columns and 132 (encodings) rows for ['encoded_tokens']
    df = pd.DataFrame(data=embeddings)
    print(len(df))
    print(df.head())
    return df


dataframe = openai_pdf_embeddings()
# map_tokens_df()
# sentence_df()
pca_kmeans_dimensionality(dataframe)
