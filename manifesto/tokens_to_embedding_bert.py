import os
import d6tflow.tasks
import pandas as pd
from transformers import BertModel, AutoModel, BertTokenizer, AutoModelForTokenClassification
from dotenv import load_dotenv
import pickle as pkl

# Load environment variables
load_dotenv()
transcript = os.getenv('TRANSCRIPT')
pdf_file_path = os.getenv('PDF_FILE_PATH')

# Embed YouTube video transcript and PDF file
pdf_text_string = str()
# extract_pdf_text = extract_pdf_text(pdf_file_path).write('manifesto_bert.txt')
# Convert pdf_text
# o string
with open('manifesto.txt', 'r') as f:
    pdf_text = f.read()
    for text in pdf_text:
        pdf_text_string += text

# Load BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load other models and tokenizers
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
tokenizer = AutoModelForTokenClassification.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")


class BertEmbeddingsTask(d6tflow.tasks.TaskPickle):
    def run(self):
        with open(transcript, 'rb') as f:
            data = pkl.load(f)
            data = [d['text'] for d in data]
            # Generate BERT embeddings for transcript
            encoding = bert_tokenizer.encode_plus(data, add_special_tokens=True, truncation=True, padding="max_length",
                                                  return_attention_mask=True, return_tensors="pt")["input_ids"]
            # transcript_input = tf.constant(transcript_tokens)[None, :]
            transcript_embeddings = bert_model(encoding)[0][0, :, :]

            # Generate BERT embeddings for PDF text
            encoding = bert_tokenizer.encode_plus(pdf_text_string, add_special_tokens=True, truncation=True, padding="max_length",
                                                  return_attention_mask=True, return_tensors="pt")["input_ids"]
            # pdf_input = tf.constant(encoding)[None, :]
            pdf_embeddings = bert_model(encoding)[0][0, :, :]
            # print(pdf_input)
            # pdf_embeddings = bert_model(pdf_input)
            # input_shape = encoding.shape

            # Output embeddings as pandas dataframe
            df = pd.DataFrame({
                'transcript_bert_embeddings': [transcript_embeddings.detach().numpy()],
                'pdf_bert_embeddings': [pdf_embeddings.detach().numpy()],
            })
            data = {'model': df}
            f.close()
        self.save(data)


flow = d6tflow.Workflow()
flow.run(BertEmbeddingsTask)