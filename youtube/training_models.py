import d6tflow
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


class TrainModelTask(d6tflow.tasks.TaskPickle):  # Assuming you have a defined Task
    def requires(self):
        return DataTask()

    def run(self):
        data = self.inputLoad()  # Load your data
        # Select only the rows where cluster_label equals 0
        data_zero = data[data['cluster_label'] == 0]

        # Preprocessing
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data_zero['text'])

        sequences = tokenizer.texts_to_sequences(data_zero['text'])
        vocab_size = len(tokenizer.word_index) + 1

        # Create input sequences and the corresponding outputs
        input_sequences = []
        output_sequences = []

        for sequence in sequences:
            for i in range(1, len(sequence)):
                input_sequences.append(sequence[:i])
                output_sequences.append(sequence[i])

        # Padding
        max_sequence_length = max([len(x) for x in input_sequences])
        input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length-1, padding='pre')
        output_sequences = tf.keras.utils.to_categorical(output_sequences, num_classes=vocab_size)

        # Define the model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 256, input_length=max_sequence_length-1),
            tf.keras.layers.LSTM(200),
            tf.keras.layers.Dense(vocab_size, activation='softmax')
        ])

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        history = model.fit(input_sequences, output_sequences, epochs=100)

        # Save the model
        self.save(history.model)
