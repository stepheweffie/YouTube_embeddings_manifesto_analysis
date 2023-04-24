from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd


# Load serialized transcript into a pandas DataFrame

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define a function to count the number of biased terms in a given .txt


def count_biased_terms(text):
    # Apply sentiment analysis to the text
    sentiment = sia.polarity_scores(text)
    # Extract the words that contributed to the positive or negative sentiment
    words = []
    for word in text.split():
        score = sia.polarity_scores(word)
        if score['pos'] > 0.5 or score['neg'] > 0.5:
            words.append(word)
    # Return the count of biased terms
    return len(words)

# Apply the count_biased_terms function to the transcript


text_file['biased_term_count'] = text_file['text'].apply(count_biased_terms)
