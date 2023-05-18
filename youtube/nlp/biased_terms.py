from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
nlp = spacy.load('en_core_web_sm')

# Load serialized transcript into a pandas DataFrame
# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()
# Define a function to count the number of biased terms in a given .txt


biased_terms = set()
has_negative_bias = list()


def nlp_biased_terms(text):

    docs = nlp.pipe(text)
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
    return terms

