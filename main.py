import spacy
from dotenv import load_dotenv
import pandas as pd
import d6tflow
import os
# from nlp.biased_terms import count_biased_terms
load_dotenv()
nlp = spacy.load('en_core_web_sm')


# Define the d6tflow task for loading the text data
class LoadChannelTranscriptsTask(d6tflow.tasks.TaskPqPandas):
    filepath = d6tflow.Parameter(default='data.csv')
    def requires(self):
        return os.getenv('YOUTUBE_CHANNEL_DATA_DIR')


    def run(self):
        df = pd.read_csv(self.filepath)
        self.save(df)


class CountBiasedTermsTask(d6tflow.tasks.TaskPickle):
    def requires(self):

    def run(self):

# TODO Arge parse for manifesto, channel transcript(s), or both analysis with d6tflow pipelines

