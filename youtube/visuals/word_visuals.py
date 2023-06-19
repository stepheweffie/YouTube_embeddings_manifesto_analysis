import pandas as pd
import nltk
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

df = pd.read_pickle(f'/Users/savantlab/PycharmProjects/YouTube_embeddings_manifesto_analysis/'
                    f'youtube/data/TrainingPreprocessTask/TrainingPreprocessTask__99914b932b-data.pkl')

df_playlist = pd.DataFrame(df['df'])
print(df_playlist['text'][:10])
# add more stop words if needed
stopwords = set(STOPWORDS)
df = df_playlist
# Tokenize the words
all_words = nltk.word_tokenize(' '.join(df['text']))
# filter stop words
all_words = [word.lower() for word in all_words if word not in stopwords]
all_words = [word for word in all_words if len(word) > 6]
word_str = ' '.join(all_words)

# Count the word frequencies
word_freq = Counter(all_words)
# Get the 10 most common words
common_words = word_freq.most_common(20)
# Create a bar plot
words, frequencies = zip(*common_words)
y_pos = range(len(words))
plt.bar(y_pos, frequencies, align='center', alpha=0.5)

# To put labels inside the bars, we'll use the 'text' function
for i in range(len(frequencies)):
    plt.text(x=i, y=frequencies[i], s=words[i], size=9, rotation=90, ha='right', va='bottom')
plt.xticks([])
plt.show()

wordcloud = WordCloud(stopwords=stopwords).generate(word_str)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()