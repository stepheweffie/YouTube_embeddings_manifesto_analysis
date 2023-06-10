import pandas as pd
import matplotlib.pyplot as plt
from main import nlp
# Load the biased terms and transcript data
biased_terms = pd.read_csv('biased_terms.csv')['biased_terms']
transcripts = pd.read_csv('transcripts.csv')

# Define a function to count the number of biased terms in a given transcript


def count_biased_terms(text):
    doc = nlp(text)
    return sum([token.lower_ in biased_terms for token in doc])

# Use pandas apply() function to apply the count_biased_terms function to each transcript


biased_counts = transcripts['texts'].apply(count_biased_terms)

# Plot the histogram of biased term counts
plt.hist(biased_counts, bins=range(11), align='left', rwidth=0.8)
plt.xticks(range(11))
plt.xlabel('Number of Biased Terms')
plt.ylabel('Frequency')
plt.title('Histogram of Biased Term Counts per Transcript')
plt.show()
