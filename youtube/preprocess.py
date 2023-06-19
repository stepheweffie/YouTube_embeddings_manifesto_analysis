import glob
import d6tflow
from d6tflow import Workflow
import os
import pickle
import pandas as pd
import spacy
from keras.preprocessing.text import text_to_word_sequence
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# If you haven't downloaded these NLTK resources yet, you will need to do so
from nltk import sent_tokenize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
# Load the dataset of your personal writing to train a model to write in your style
data_dir = "data"
files = os.listdir(data_dir)
channel_name = '@JordanBPeterson'
sia = SentimentIntensityAnalyzer()


def biased_score(word_or_text):
    biases = {}
    score = sia.polarity_scores(word_or_text)
    biases[word_or_text] = score
    return biases


class PicklesToDFTask(d6tflow.tasks.TaskCachePandas):
    # Cache the original transcripts in a dict of dataframes
    def run(self):
        output = {}
        pkl_files = glob.glob(f'{data_dir}/{channel_name}/*.pkl')
        for pkl_file in pkl_files:
            try:
                df = pd.read_pickle(pkl_file)
                df = pd.DataFrame(df)
                # Perform some operations on the dataframe
                output[pkl_file] = df
                # Save the processed dataframe as output
            except FileNotFoundError:
                continue
        self.save(output)


class TrainingPreprocessTask(d6tflow.tasks.TaskPickle):
    # Cache the original transcripts in a dict of dataframes
    def run(self):
        texts = list()
        tokens = list()
        rows = list()
        text_tokens = list()
        pkl_files = glob.glob(f'{data_dir}/playlist/*.pkl')
        for pkl_file in pkl_files:
            try:
                data = nltk.data.load(pkl_file, format='pickle')
                # Perform some operations on the dataframe
                for text in data:
                    text_tokens.append(tokens)
                    if len(text['text']) > 1:
                        texts = text['text']
                        tokens = word_tokenize(text['text'])
                row = {'video_id': pkl_file, 'text': texts, 'tokens': tokens}
                rows.append(row)
            except FileNotFoundError:
                continue
            except KeyError:
                print("KeyError occurred, Check Luigi For Status")
        rows = pd.DataFrame(rows)
        df = {'df': rows}
        print(df)
        self.save(df)


class SearchDfTask(d6tflow.tasks.TaskPickle):

    def requires(self):
        return PicklesToDFTask()

    def run(self):
        dataframes = self.inputLoad()
        results = {}
        search_substring = input('Give a search string: ')
        output_dir = "search_" + search_substring
        # Iterate through the dictionary of dataframes
        for filename, df in dataframes.items():
            # Apply the search operation to the 'A' column of each dataframe
            search_results = df[df['text'].str.contains(search_substring, case=False)]
            # Print the filename and the rows containing the substring
            if not search_results.empty:
                print(f"Results in {filename}:")
                results[f'{filename}'] = search_results
                print(search_results)
                # If you want to access the whole row as a pandas Series
                for _, row in search_results.iterrows():
                    print(row)
                print('\n')
        os.mkdir(output_dir)
        os.chdir(output_dir)
        self.save(results)


# Define the d6tflow task for loading the text data and labels
class LoadDataTask(d6tflow.tasks.TaskPickle):

    def run(self):
        # Define a list of text data and corresponding labels
        data = []
        transcripts = os.listdir(f'{data_dir}/{channel_name}')
        for video_id in transcripts:
            if video_id.endswith(".pkl"):
                with open(os.path.join(f"{data_dir}/{channel_name}", video_id), "rb") as f:
                    file_array = pickle.load(f)
                    text_list = [d["text"] for d in file_array]
                    text = ','.join(text_list)
                    new_dict = {"text": text, "label": video_id}
                    data.append(new_dict)
        # Save the data as a dictionary
        data = {'text': [d['text'] for d in data],
                'label': [d['label'] for d in data]}
        self.save(data)


# Define the d6tflow task for preprocessing the text data
class BiasedSentencesTask(d6tflow.tasks.TaskPqPandas):
    persist = ['video_transcript_data', 'biased_sentences_data']

    def requires(self):
        return LoadDataTask()

    def run(self):
        # Load the text data and labels
        data = self.inputLoad()
        sentences = []
        text = data['text']
        labels = data['label']
        df = pd.DataFrame()
        df['video ids'] = labels
        words = []
        doc_sents = []
        # Get the biased words and sentences in lists of dicts of strings
        biased_sents = {}
        unbiased_sents = {}
        # Keras text_to_word_sequence for sentence tokenization
        sentences.append(text_to_word_sequence(text, split=".", filters="!,?.\n"))
        words.append([word_tokenize(text)])
        for sentence_list in sentences:
            for sentence in sentence_list:
                # Call biased_score on each token sentence
                biased = biased_score(sentence)
                for val in biased.values():
                    if val['neg'] or val['pos'] >= 0.5:
                        biased_sents[sentence] = val
                    else:
                        unbiased_sents[sentence] = val
        doc_sents.append([biased_sents, unbiased_sents])
        # Assuming there are biased words and sentences in every transcript (long form)
        df['word_tokens'] = words
        # because = df['word_tokens'].str.contains('because')
        df['sent_tokens'] = doc_sents
        sentences = pd.DataFrame(df['sent_tokens']).transpose()
        self.save({'video_transcript_data': df,
                   'biased_sentences_data': sentences})


# Define a list of stop words and punctuation to remove from the text data
# stop_words = stopwords.words('english') + list(string.punctuation)


if __name__ == '__main__':
    flow = Workflow()
    # flow.run(LoadDataTask)
    # flow.run(ExtractTextAfterBecause)
    # flow.run(PreprocessDataTask)
    flow.run(TrainingPreprocessTask)
    # flow.run(PicklesToDFTask)


