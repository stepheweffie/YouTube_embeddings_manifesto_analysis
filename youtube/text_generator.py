import d6tflow
from writer_model import TrainModelTask
import tensorflow as tf


# Define the d6tflow task for generating text from the model
class GenerateTextTask(d6tflow.tasks.TaskPickle):
    def requires(self):
        return TrainModelTask()

    def run(self):
        # Load the trained model and preprocessed data
        data = self.input().load()
        X_val = data['X_val']
        label_encoder = data['label_encoder']
        vectorizer = data['vectorizer']
        model = tf.keras.models.load_model('my_model.h5')

        # Generate text using the trained model
        text_idx = np.random.randint(0, X_val.shape[0])
        input_vec = X_val[text_idx, :]
        input_text = vectorizer.inverse_transform(input_vec)[0]
        predicted_label_idx = model.predict_classes(input_vec.reshape(1, -1))[0]
        predicted_label = label_encoder.inverse_transform(predicted_label_idx)

        # Save the generated text
        generated_text = {'input_text': input_text,
                          'predicted_label': predicted_label}
        self.save(generated_text)
