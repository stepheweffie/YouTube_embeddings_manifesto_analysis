import d6tflow
from d6tflow import Workflow
# from nltk.corpus import stopwords
import os
import pickle
import pandas as pd
import spacy
from keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

nlp = spacy.load('en_core_web_sm')

# Load the dataset of your personal writing to train a model to write in your style
data_dir = "data"
files = os.listdir(data_dir)
channel_name = '@JordanBPeterson'
sia = SentimentIntensityAnalyzer()


# Define the d6tflow task for loading the text data and labels
class LoadDataTask(d6tflow.tasks.TaskPickle):

    def run(self):
        # Define a list of text data and corresponding labels
        print(channel_name, data_dir)
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


class PicklesToDFTask(d6tflow.tasks.TaskCachePandas):
    def run(self):
        output = {}
        pkl_files = os.listdir(f'{data_dir}/{channel_name}')
        os.chdir(f'{data_dir}/{channel_name}')
        for pkl_file in pkl_files:
            df = pd.read_pickle(pkl_file)
            df = pd.DataFrame(df)
            # Perform some operations on the dataframe
            output[pkl_file] = df
            # Save the processed dataframe as output
        self.save(output)


def biased_score(word_or_text):
    biases = {}
    score = sia.polarity_scores(word_or_text)
    biases[word_or_text] = score
    return biases


# Define the d6tflow task for preprocessing the text data
class PreprocessDataTask(d6tflow.tasks.TaskPqPandas):
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
        # Locate a video_id
        # print(df['video ids'].str.contains(video_id))
        words = []
        doc_sents = []
        # Get the biased words and sentences in lists of dicts of strings
        # for doc in text:
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


class ExtractTextBecause(d6tflow.tasks.TaskPqPandas):
    def requires(self):
        return PreprocessDataTask()

    def run(self):
        data = self.inputLoad(as_dict=True)['biased_sentences_data']
        because_bias = []
        for row in data:
            because = []
            # because_str = data.str.contains('because')
            for sent_list in row:
                for sent_dict in sent_list:
                    sent = 'because'
                    for sents in sent_dict.keys():
                        if sent in sents:
                            because.append(sent)
                        else:
                            because.append(' ')
            because_bias.append(because)
        output = pd.DataFrame()
        output['because_bias'] = because_bias
        data = output.to_dict()
        self.save(data)


if __name__ == '__main__':
    flow = Workflow()
    # flow.run(LoadDataTask)
    # flow.run(ExtractTextAfterBecause)
    # flow.run(PreprocessDataTask)
    flow.run(PicklesToDFTask)
    data = flow.outputLoad(PicklesToDFTask)



