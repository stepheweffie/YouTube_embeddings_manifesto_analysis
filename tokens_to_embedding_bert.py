import os
import pandas as pd
import tensorflow as tf
from transformers import TFBertTokenizer, TFBertModel, TFAutoModelForTokenClassification, TFAutoModel
from dotenv import load_dotenv
from manifesto_data import extract_pdf_text
load_dotenv()



# Load BERT model and tokenizer
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = TFBertTokenizer.from_pretrained('bert-base-uncased')

# Load other models and tokenizers
model = TFAutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
tokenizer = TFAutoModelForTokenClassification.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

# Embed YouTube video transcript and PDF file
pdf_file_path = os.environ['PDF_FILE_PATH']
pdf_text = extract_pdf_text(pdf_file_path)

# Generate BERT embeddings for transcript
transcript_tokens = bert_tokenizer.encode(transcript, add_special_tokens=True)
transcript_input = tf.constant(transcript_tokens)[None, :]
transcript_embeddings = bert_model(transcript_input)[0][0, :, :]

# Generate sentence embeddings for transcript using other model
transcript_sentence_embeddings = model.encode(transcript, tokenizer=tokenizer)

# Generate BERT embeddings for PDF text
pdf_tokens = bert_tokenizer.encode(pdf_text, add_special_tokens=True)
pdf_input = tf.constant(pdf_tokens)[None, :]
pdf_embeddings = bert_model(pdf_input)[0][0, :, :]

# Generate sentence embeddings for PDF text using other model
pdf_sentence_embeddings = model.encode(pdf_text, tokenizer=tokenizer)

# Output embeddings as pandas dataframe
df = pd.DataFrame({
    'transcript_bert_embeddings': [transcript_embeddings.numpy()],
    'transcript_sentence_embeddings': [transcript_sentence_embeddings.tolist()],
    'pdf_bert_embeddings': [pdf_embeddings.numpy()],
    'pdf_sentence_embeddings': [pdf_sentence_embeddings.tolist()]
})

if __name__ == '__main__':
    import subprocess
    subprocess.run('tokens_to_embedding_bert.py')
    subprocess.run('plot.py')