import spacy
from dotenv import load_dotenv
import pandas as pd
import d6tflow
import os
from youtube.preprocess import PreprocessDataTask
# from nlp.biased_terms import count_biased_terms
load_dotenv()
nlp = spacy.load('en_core_web_sm')

