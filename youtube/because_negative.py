import pandas as pd
import d6tflow
from main import nlp
# Load serialized transcripts into a pandas DataFrame
# Define the list of biased terms
biased_terms = set()
has_negative_bias = list()


class UnPickleTranscripts(d6tflow.tasks.TaskPqPandas):
    def requires(self):
        # Returns a list of pickles
        return LoadTranscripts()

    def run(self):
        transcripts = self.input().load()
        for transcript in transcripts:
            pd.read_pickle(transcript)
        df = pd.DataFrame(data=transcripts)
        self.save(df.to_dict())

# Define a task to extract the biased terms from the transcripts


class ExtractBiasedTerms(d6tflow.tasks.TaskPqPandas):
    def requires(self):
        return LoadTranscripts()

    def run(self):
        transcripts = self.input().load()
        # Define a spaCy pipeline to process the transcripts
        docs = nlp.pipe(transcripts['text'])
        # Extract the biased terms from each transcript using spaCy's dependency parser
        for doc in docs:
            for token in doc:
                if token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass':
                    if token.head.text.lower() == 'not' or 'n\'t' in token.head.text.lower():
                        for child in token.children:
                            if child.lower_ not in biased_terms:
                                biased_terms.add(child.lower_)
                    elif token.lower_ not in biased_terms:
                        biased_terms.add(token.lower_)
        terms = {'biased_terms': list(biased_terms)}
        self.save(terms)

# Define a function to extract the text after the word 'because' in a given sentence

# Define a function to determine whether a sentence contains negative bias towards certain terms
def negative_bias(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.lower_ in biased_terms and (token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass'):
            if token.head.text.lower() == 'not' or 'n\'t' in token.head.text.lower():
                return True
    return False

# Define the workflow


class DetermineNegativeBias(d6tflow.tasks.TaskPqPandas):

    def requires(self):
        return {'text': ExtractTextAfterBecause(), 'biased_terms': ExtractBiasedTerms()}

    def run(self):
        transcripts = self.input()['text'].load()
        # Define a spaCy pipeline to process the transcripts
        docs = nlp.pipe(transcripts['text_after_because'])
        # Determine whether each sentence contains negative bias towards certain terms
        for doc in docs:
            if doc is None:
                has_negative_bias.append(False)
            else:
                found_bias = False
                for token in doc:
                    if token.lower_ in biased_terms and (token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass'):
                        if token.head.text.lower() == 'not' or 'n\'t' in token.head.text.lower():
                            found_bias = True
                has_negative_bias.append(found_bias)
        transcripts['has_negative_bias'] = has_negative_bias
        self.save(transcripts)
