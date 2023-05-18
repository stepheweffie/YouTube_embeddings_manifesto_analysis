import numpy as np
import pickle
from preprocess import PreprocessDataTask
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, MaxPooling1D, Embedding
from keras.optimizers import Adam
import d6tflow


x_train = []
y_train = []


# Define the d6tflow task for training the model
class TrainModelTask(d6tflow.tasks.TaskPickle):
    def requires(self):
        return PreprocessDataTask()

    def run(self):
        # Load the preprocessed data
        data = self.input().load()
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']

        # Define the model architecture
        model = Sequential()
        model.add(Embedding(5000, 64, input_length=100))
        model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(LSTM(64))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dense(np.unique(y_train).size, activation='softmax'))

        # Compile the model
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=1,validation_split=0.2, epochs=10,
                  batch_size=32)

        # Save the trained model and preprocessed data
        trained_data = {'model': model,
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_val': X_val,
                        'y_val': y_val,
                        'label_encoder': label_encoder,
                        'vectorizer': vectorizer}
        # Evaluate the model
        test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)
        print('\nTest accuracy:', test_acc)
        self.save(trained_data)


def generate_text_and_save(seed_text, filename):
    generated_text = seed_text
    while True:
        predictions = history.predict(x_train, verbose=0)[0]
        predicted_word_index = np.argmax(predictions)
        predicted_word = tokenizer.index_word[predicted_word_index]
        generated_text += " " + predicted_word
        if predicted_word == "." or len(generated_text) >= 1000:
            break
    with open(f"data/{channel_name}/text/{filename}.pkl", "wb") as f:
        pickle.dump(generated_text, f)

    return f"data/text/{filename}.txt"


