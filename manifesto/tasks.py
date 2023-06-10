from manifesto_openai_embeddings import openai_pdf_embeddings
import d6tflow
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch


# The first two embeddings tasks use OpenAI embeddings-ada-002 model

class SingleVideoEmbeddingsTask(d6tflow.tasks.TaskPickle):
    def run(self):
        # Currently subprocess must be called on the file prior; single_video_openai_embeddings.py
        video_embeddings = pd.read_pickle('data/single_video_openai_embeddings.pkl')
        df = pd.DataFrame()
        print(video_embeddings)
        data = {'video_embeddings': df}
        self.save(data)


class PDFEmbeddingsTask(d6tflow.tasks.TaskPickle):
    def run(self):
        pdf_embeddings = openai_pdf_embeddings()
        df = pd.DataFrame(pdf_embeddings)
        data = {'pdf_embeddings': df}
        self.save(data)


class BERTEmbeddingsTask(d6tflow.tasks.TaskPickle):
    document_path = d6tflow.Parameter()

    def run(self):
        # Load pre-trained model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')

        with open(self.document_path, 'r') as file:
            text = file.read().replace('\n', '')

        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt")

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Take mean of token-level embeddings to get sentence-level embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)

        self.save(embeddings.numpy())


flow = d6tflow.Workflow()
flow.run(PDFEmbeddingsTask)
