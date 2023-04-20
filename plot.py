import seaborn as sns
import matplotlib.pyplot as plt
from tokens_to_embedding_bert import extract_pdf_text, extract_youtube_transcript, df, pdf_file_path, youtube_video_id
import numpy as np
import pandas as pd
# Assuming that the `df` dataframe has already been generated

# Unpack the embeddings from the dataframe
transcript_bert_embeddings = df['transcript_bert_embeddings'].iloc[0]
transcript_sentence_embeddings = df['transcript_sentence_embeddings'].iloc[0]
pdf_bert_embeddings = df['pdf_bert_embeddings'].iloc[0]
pdf_sentence_embeddings = df['pdf_sentence_embeddings'].iloc[0]

# Concatenate the embeddings
all_embeddings = np.concatenate([
    transcript_bert_embeddings,
    transcript_sentence_embeddings,
    pdf_bert_embeddings,
    pdf_sentence_embeddings
], axis=0)

# Create labels for the embeddings
labels = (['Transcript (BERT)'] * 768
          + ['Transcript (Sentence)'] * 768
          + ['PDF (BERT)'] * 768
          + ['PDF (Sentence)'] * 768)

# Unpack the transcript and PDF text
transcript = extract_youtube_transcript(youtube_video_id)
pdf_text = extract_pdf_text(pdf_file_path)

# Split the transcript and PDF text into sentences
transcript_sentences = transcript.split('.')
pdf_sentences = pdf_text.split('.')

# Create a list of all the sentences
all_sentences = (transcript_sentences
                 + transcript_sentences
                 + pdf_sentences
                 + pdf_sentences)

# Convert the embeddings and sentences to a pandas dataframe
embedding_df = pd.DataFrame(all_embeddings, columns=['x', 'y'])
embedding_df['label'] = labels
embedding_df['sentence'] = all_sentences

# Create a scatter plot of the embeddings
sns.scatterplot(data=embedding_df, x='x', y='y', hue='label')

# Add text labels to the plot
for i in range(len(all_sentences)):
    plt.annotate(all_sentences[i], (all_embeddings[i, 0], all_embeddings[i, 1]))

# Show the plot
plt.show()
