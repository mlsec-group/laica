import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment.experiment import Experiment
from model.safe_embedding import SAFEEmbedding


class FunctionSimilarityExperiment(Experiment):
    def __init__(self, embedding_model, max_length, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self.max_length = max_length
        self.embedding_size = embedding_model.get_embedding_size()

    def embedding(self, function):
        embedding = self.embedding_model.get_embedding(function)
        padded = self.embedding_model.pad_embedding(embedding, self.max_length, self.embedding_size)
        return padded.astype(np.float32)

    def prepare_ds(self, split):
        dataset = self.dataset[split]

        def gen():
            for pair in dataset:
                yield (
                    (self.embedding(pair["function_0"]), self.embedding(pair["function_1"])),
                    [float(pair["label"])],
                )

        embedding_spec = tf.TensorSpec(shape=(self.max_length, self.embedding_size), dtype=tf.float32)

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                (embedding_spec, embedding_spec),
                tf.TensorSpec(shape=(1,), dtype=tf.float32),
            ),
        )

    def validation_metric(self):
        return "val_auc"

    def adapt_data_encoder(self, raw_train_ds):
        pass

    def encode_ds(self, ds):
        return ds

    def save_vecorizer(self):
        pass

    def load_vectorizer(self):
        pass

    def compile_model(self, model):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.initial_learning_rate),
            loss='mse',
            metrics='AUC',
        )

    def cosine_similarity(self, embeddings):
        x = tf.math.l2_normalize(embeddings[0], axis=1)
        y = tf.math.l2_normalize(embeddings[1], axis=1)
        return tf.clip_by_value(tf.reduce_sum(tf.multiply(x, y), axis=1), -1.0, 1.0)

    def scale_cosine_similarity(self, similarity):
        return (similarity + 1) / 2

    def prepare_model(self):
        inputs = tf.keras.Input(shape=(None, self.embedding_size), name='function')
        embedding = SAFEEmbedding()(inputs)

        function_embedding = tf.keras.models.Model(
            inputs=[inputs],
            outputs=[embedding],
            name="safe_function_embedding",
        )

        input_0 = [tf.keras.Input(shape=i.shape[1:], dtype=i.dtype, name=f"{i.name}0")
                   for i in function_embedding.inputs]
        input_1 = [tf.keras.Input(shape=i.shape[1:], dtype=i.dtype, name=f"{i.name}1")
                   for i in function_embedding.inputs]

        similarity = tf.keras.layers.Lambda(self.cosine_similarity, name="similarity")
        scale = tf.keras.layers.Lambda(self.scale_cosine_similarity, name="label")

        output = scale(similarity((function_embedding(input_0), function_embedding(input_1))))

        siamese_model = tf.keras.models.Model(
            inputs=input_0 + input_1,
            outputs=[output],
            name="siamese_network",
        )
        self.compile_model(siamese_model)

        return siamese_model

    def predictions(self, model, ds):
        return model.predict(ds).reshape(-1)

    def labels(self, ds):
        return np.array([label.numpy()[0] for data, label in ds.unbatch()])

    def metrics(self, y_true, y_pred):
        return {
            "auc": roc_auc_score(y_true, y_pred),
            "average_precision": average_precision_score(y_true, y_pred),
        }
