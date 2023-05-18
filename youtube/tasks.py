import numpy as np
import pandas as pd
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sia = SentimentIntensityAnalyzer()
# Load the dataset of your personal writing to train a model to write in your style
data_dir = "data"
files = os.listdir(data_dir)
channel_name = '@JordanBPeterson'
print(channel_name)

# LoadDataTask
# PreprocessDataTask

# BiasedTextTask
# GenerateBiasedTextTask

# Train a language generation model on the biased text

# pad the padded sequences with zeros to reach the corpus length of 561
padded_sequences = pad_sequences(padded_sequences, maxlen=561, padding='post', truncating='post')

# Define the sequence model
sequence_input = Input(shape=(maxlen,))
embedded_sequences = Embedding(12, 8)(sequence_input)
lstm = LSTM(32)(embedded_sequences)

# Bag-of-words
vectorizer = CountVectorizer()
text = text['text']
X = vectorizer.fit_transform(text)
print(type(X[0]), type(X[1]))

# Define the input and output shapes for the Keras model
input_shape = X.shape[1]
output_shape = len(tokenizer.word_index) + 1

# Assuming `labels` is a list of strings
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
encoded_labels.astype(np.float32)
# Determine the number of classes
num_classes = len(le.classes_)

# Create an empty array for y
y = np.zeros((X.shape[0], num_classes)).astype(np.float32)

# Fill in the correct label for each sample in y
for i in range(X.shape[0]):
    # Get the label for this sample
    label = encoded_labels[i]
    # Set the corresponding value in y to 1
    y[i, label] = 1

# Define the bag-of-words model
bow_input = Input(shape=(X.shape[0],))
dense = Dense(32)(bow_input)

# Concatenate the two models
merged = concatenate([lstm, dense])
output = Dense(output_shape, activation='sigmoid')(merged)

# Define the combined model
model = Model(inputs=[sequence_input, bow_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Keras model on the tokenized and sequenced data
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

# Create a sparse tensor from the bag-of-words representation
indices = tf.cast(tf.transpose([X.nonzero()[0], X.nonzero()[1]]), tf.int64)
values = tf.cast(X.data, tf.float32)
shape = tf.cast(X.shape, tf.int64)

# Check types of NumPy arrays
sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

# Reorder the indices of the sparse tensor
sparse_tensor_reordered = tf.sparse.reorder(sparse_tensor)

# Convert the sparse tensor to a dense tensor
dense_tensor = tf.sparse.to_dense(sparse_tensor_reordered)

# Convert the input data to NumPy arrays
padded_sequences = np.array(padded_sequences)
padded_sequences = np.vstack([np.zeros((209, 561)).astype(np.float32), padded_sequences]).astype(np.float32)
# print(padded_sequences)
# Transform sentiment
sentiment = df
sentiment = sentiment.iloc[:561, :]
sentiment = sentiment.to_numpy()
# Fit the dense_tensor to the number of documents
dense_tensor = np.asarray(dense_tensor).astype(np.float32)

Y = y

[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
[print(l.name, l.input_shape, l.dtype) for l in model.layers]

# define batch size
batch_size = 32
X = np.concatenate(padded_sequences).astype(np.float32)
# Compile and fit the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# call model.fit()
model.fit(x=[padded_sequences.astype(np.float32), dense_tensor.astype(np.float32)], y=Y.astype(np.float32),
          batch_size=batch_size)

topic = "feminism"
topic_vector = vectorizer.transform([topic]).toarray()
seed_text = "Tell me something that is happening because of feminism"
for i in range(20):
    encoded_seed = tokenizer.texts_to_sequences([seed_text])[0]
    encoded_seed = pad_sequences([encoded_seed], maxlen=1, padding='pre')
    predicted_word = model.predict([topic_vector], verbose=0)
    for word, index in tokenizer.word_index.items():
        if index == np.argmax(predicted_word):
            seed_text += " " + word
            break

        '''
        maxlen = max([len(seq) for seq in biased_sequences])
        padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')
        vocab_size = len(tokenizer.word_index) + 1
        # Vectorize text and render vocabulary 
        vectorizer = CountVectorizer()
        vectorizer.fit(text)
        X = vectorizer.fit_transform(text)
        print(X, vectorizer.vocabulary)

        # Encode the labels using a LabelEncoder object
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)

        sequences = np.array(sequences)
        X_train, y_train = sequences[:, :-1], sequences[:, -1]

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save the preprocessed data
        preprocessed_data = {'X_train': X_train,
                             'X_val': X_val,
                             'y_train': y_train,
                             'y_val': y_val,
                             'vectorizer': vectorizer,
                             'label_encoder': label_encoder,
                             'sequences': padded_sequences,
                             'vocab_size': vocab_size}
        self.save(preprocessed_data)
'''