import pandas as pd
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelSpreading
import d6tflow
from d6tflow import Workflow
from preprocess import TrainingPreprocessTask
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# If you haven't downloaded these NLTK resources yet, you will need to do so
# nltk.download('vader_lexicon')  # for sentiment analysis
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

sia = SentimentIntensityAnalyzer()
data_dir = "data"
files = os.listdir(data_dir)
channel_name = '@JordanBPeterson'

# Load the dataset of your personal writing to train a model to write in your style


class TrainingTask(d6tflow.tasks.TaskPqPandas):
    def requires(self):
        return TrainingPreprocessTask()

    def run(self):
        self.save(output)


class BinaryTrainingTask(d6tflow.tasks.TaskPqPandas):  # TaskPqPandas is a task that loads/saves dataframes in parquet format
    def requires(self):
        return TrainingTask()

    def run(self):
        # Load your data
        data_dict = self.inputLoad()
        vectorizer = TfidfVectorizer(stop_words='english')
        for key in data_dict:
            df = data_dict[key]
            print(key)
            # First 10 rows from each dataframe are from 'main speaker' (labeled as 0)
            # Rest are uncertain (labeled as -1)
            df['label'] = [-1 for _ in range(len(df))]  # Initialize all as -1 (uncertain)
            df.iloc[:20, df.columns.get_loc('label')] = 0  # First 10 rows are 'main speaker'
            # Convert the text to a matrix of TF-IDF features
            X = vectorizer.fit_transform(df['text'])
            # Convert sparse matrix to dense matrix
            X = X.toarray()
            # Run K-means clustering to identify groups
            kmeans = KMeans(n_clusters=2)  # Choose the appropriate number of clusters
            kmeans.fit(X)
            # Now each document in your corpus has a cluster label, which is stored in kmeans.labels_
            df['cluster_label'] = kmeans.labels_
            # If you want to see the terms that are most indicative of each cluster, you could do something like this:
            print("Top terms per cluster:")
            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names_out()
            for i in range(2):  # Again, replace 5 with your chosen number of clusters
                print("Cluster %d:" % i)
                for ind in order_centroids[i, :10]:  # Replace 10 with the number of terms you want to print out
                    print(' %s' % terms[ind])
                print(silhouette_score(X, kmeans.labels_))
            print(X.shape)
            # Convert labels
            y = df['label']
            # Use LabelSpreading to propagate the labels from the labeled instances to the unlabeled ones
            label_propagation_model = LabelSpreading(kernel='knn', alpha=0.8)
            try:
                label_propagation_model.fit(X, y)
                print(label_propagation_model.score(X, y))
                # Predict labels for the unlabeled data
            except ValueError:
                print('ValueError')
                continue
            predicted_labels = label_propagation_model.transduction_
            # Replace the uncertain labels in your dataframe with the predicted labels
            df['label'] = predicted_labels
        # Combine all dataframes into one
        data = pd.concat(data_dict.values())
        # Save the dataframe
        self.save(data)


class BinaryClassifierTask(d6tflow.tasks.TaskPickle):  # Assuming you have a defined Task
    def requires(self):
        return BinaryTrainingTask()

    def run(self):
        data = self.inputLoad()  # Load your data

        # Tokenization
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(data['text'])
        sequences = tokenizer.texts_to_sequences(data['text'])

        # Padding
        padded = pad_sequences(sequences, padding='post')

        # Splitting the data into training and testing
        train_size = int(0.8 * len(padded))  # Adjust as needed
        train_sequences = padded[:train_size]
        train_labels = data['cluster_label'][:train_size]
        test_sequences = padded[train_size:]
        test_labels = data['cluster_label'][train_size:]

        # Define the model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 64, input_length=train_sequences.shape[1]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        history = model.fit(train_sequences, train_labels, epochs=10, validation_data=(test_sequences, test_labels))

        # Save the model
        self.save(history.model)


flow = Workflow()
flow.run(TrainingTask)
# output = flow.outputLoad(DataTask)
# print(output.head())