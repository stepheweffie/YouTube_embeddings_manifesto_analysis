import numpy as np
import pickle
from preprocess import TrainingPreprocessTask
from d6tflow import Workflow
from gensim import corpora, models
from gensim.models import Phrases, CoherenceModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import d6tflow
# import cfg, tasks, visualize


class TransformerTask(d6tflow.tasks.TaskPickle):
    def requires(self):
        return TrainingPreprocessTask()

    def run(self):
        data = self.inputLoad()  # Load your data
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)

        # Tokenize your texts and prepare them as inputs to the model
        inputs = tokenizer(data['text'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Define your training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        # Define your trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=inputs,
        )

        # Train the model
        trainer.train()
        self.save(model)


class LSATask(d6tflow.tasks.TaskPickle):
    def requires(self):
        return TrainingPreprocessTask()

    def run(self):
        data = self.inputLoad()  # Load your data
        data = data['texts'].tolist()
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(data)
        lsa = TruncatedSVD(n_components=10, random_state=1).fit(tfidf)
        data = {'model': lsa}
        self.save(data)


class NMFTask(d6tflow.tasks.TaskPickle):
    def requires(self):
        return TrainingPreprocessTask()

    def run(self):
        data = self.inputLoad()
        data = data['texts'].tolist()
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(data)
        nmf = NMF(n_components=10, random_state=1).fit(tfidf)
        data = {'model': nmf}
        self.save(data)


class LDATask(d6tflow.tasks.TaskPickle):
    def requires(self):
        return TrainingPreprocessTask()  # or whatever task outputs your BoW representations

    def run(self):
        data = self.inputLoad()  # Load your data
        # A topic model for bigrams
        # Process each sentence token
        sentences = [[sent] for sent in data['texts']]
        # Train the Phrases model
        bigram = Phrases(sentences, min_count=3, threshold=25)
        # Apply the trained Phrases model to our corpus
        documents_bigrams = [bigram[sentence] for sentence in sentences]
        # Create a dictionary representation of the documents.
        dictionary = corpora.Dictionary(documents_bigrams)
        corpus = [dictionary.doc2bow(text) for text in documents_bigrams]
        # Build LDA model
        # Train the LDA model
        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=18)
        # Examine the topics
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=documents_bigrams, dictionary=dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        lda_model = {'lda_model': lda_model, 'dictionary': dictionary, 'corpus': corpus, 'num_topics': 18}
        self.save(lda_model)  # Save the model


flow = Workflow()
# flow.run(TrainModelTask)
# flow.preview(LDAModel)
# flow.reset(LDAModel)
flow.run(LDATask)
flow.run(NMFTask)


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


def lda():

    lda_model = models.LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                random_state=100,
                                update_every=1,
                                chunksize=100,
                                passes=10,
                                alpha='auto',
                                per_word_topics=True)