import csv
import json
import math
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from util import log


def split_seq(data):
    return tf_text.regex_split(data, '[|/]')


def set_memory_growth():
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def hf_to_tf_dataset(dataset):
    dataset = dataset.select_columns(["data", "label"])

    def gen():
        for sample in dataset:
            yield sample["data"], sample["label"]

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )


class Experiment:
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(
        self,
        dataset,
        output_dir,
        seed,
        dimension,
        seq_length,
        max_features,
        initial_learning_rate,
        decay_rate,
        max_epochs,
        batch_size,
        min_delta,
        patience,
    ):
        set_memory_growth()
        tf.keras.utils.set_random_seed(seed)
        self.dataset = dataset
        self.output_dir = output_dir
        self.seed = seed
        self.dimension = dimension
        self.seq_length = seq_length
        self.max_features = max_features
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.min_delta = min_delta
        self.patience = patience
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=None,
            split=split_seq,
            max_tokens=self.max_features,
            output_mode='int')

    def num_classes(self):
        raise NotImplementedError()

    def validation_metric(self):
        return "val_categorical_accuracy"

    def get_output_layers(self):
        raise NotImplementedError()

    def compile_model(self, model):
        raise NotImplementedError()

    def truncate(self, data):
        data = tf.strings.split(data, '|')[:self.seq_length]
        return tf.strings.reduce_join(data, separator='|')

    def prepare_ds(self, split):
        ds = hf_to_tf_dataset(self.dataset[split])
        return ds.map(
            lambda data, label: (self.truncate(data), label),
            num_parallel_calls=Experiment.AUTOTUNE)

    def adapt_data_encoder(self, raw_train_ds):
        log("[+] Adapting the vectorization layer")
        text_ds = raw_train_ds.map(lambda data, label: data, num_parallel_calls=Experiment.AUTOTUNE)
        self.vectorize_layer.adapt(text_ds.take(50000))
        log(f"[+] Vocabulary size: {len(self.vectorize_layer.get_vocabulary())}")

    def encode_ds(self, ds):
        def encode(data, label):
            return self.vectorize_layer(tf.expand_dims(data, -1)), tf.one_hot(label, self.num_classes())

        return ds.batch(self.batch_size).map(encode, num_parallel_calls=Experiment.AUTOTUNE).unbatch()

    def tune_ds(self, ds):
        return ds.batch(self.batch_size).prefetch(Experiment.AUTOTUNE)

    def get_input_layers(self):
        embedding_matrix = np.random.random_sample(size=(self.max_features, self.dimension))
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.max_features,
            output_dim=self.dimension,
            mask_zero=True,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            activity_regularizer=tf.keras.regularizers.L2(1e-3),
            embeddings_regularizer=tf.keras.regularizers.L2(1e-3))
        return [embedding_layer,
                tf.keras.layers.Dropout(0.2)]

    def prepare_model(self):
        model = tf.keras.Sequential(self.get_input_layers() + self.get_output_layers())
        self.compile_model(model)
        return model

    def model_save_path(self):
        return self.output_dir / 'model.final'

    def vectorizer_save_path(self):
        return self.output_dir / 'vectorizer.pkl'

    def save_vecorizer(self):
        with self.vectorizer_save_path().open("wb") as f:
            pickle.dump({'config': self.vectorize_layer.get_config(),
                         'weights': self.vectorize_layer.get_weights()}, f)

    def load_vectorizer(self):
        with self.vectorizer_save_path().open("rb") as f:
            data = pickle.load(f)
            vectorizer = tf.keras.layers.TextVectorization.from_config(data['config'])
            vectorizer.set_weights(data['weights'])
            self.vectorize_layer = vectorizer

    def save_model(self, model):
        model.save_weights(str(self.model_save_path()))
        self.save_vecorizer()

    def train_model(self, train_ds, validation_ds):
        def lr_scheduler(epoch, lr):
            return lr * self.decay_rate if epoch > 0 else lr

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=self.min_delta,
                patience=self.patience,
                verbose=1,
                restore_best_weights=True),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.CSVLogger(filename=str(self.output_dir / 'training.log')),
            tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0),
        ]

        model = self.prepare_model()
        model.summary()
        model.fit(
            train_ds,
            validation_data=validation_ds,
            verbose=1,
            epochs=self.max_epochs,
            callbacks=callbacks)

        return model

    def train(self):
        raw_train_ds = self.prepare_ds("train")
        raw_validation_ds = self.prepare_ds("validation")

        self.adapt_data_encoder(raw_train_ds)

        train_ds = self.tune_ds(self.encode_ds(raw_train_ds))
        validation_ds = self.tune_ds(self.encode_ds(raw_validation_ds))

        log("[+] Training the model")
        model = self.train_model(train_ds, validation_ds)
        self.save_model(model)

        return model

    def predictions(self, model, ds):
        return np.argmax(model.predict(ds), axis=-1)

    def labels(self, ds):
        return np.array([np.argmax(label) for data, label in ds.unbatch()])

    def metrics(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }

    def write_results(self, y_true, y_pred, params):
        with (self.output_dir / 'predictions.csv').open("wt") as f:
            for true, pred in zip(y_true, y_pred):
                print(true, pred, file=f)

        metrics = self.metrics(y_true, y_pred)
        (self.output_dir / 'metrics.json').write_text(json.dumps({**metrics, **params}, indent=2))
        log(f"[+] Metrics: {metrics}")

        return metrics

    def write_objective(self, params):
        metric = self.validation_metric()

        best = None
        with (self.output_dir / 'training.log').open() as f:
            for row in csv.DictReader(f):
                val_loss = float(row["val_loss"])
                if math.isnan(val_loss):
                    continue
                if best is None or val_loss < best[0]:
                    best = (val_loss, float(row[metric]), int(row["epoch"]))

        objective = {
            "objective": 0.0 if best is None else best[1],
            "metric": metric,
            "epoch": None if best is None else best[2],
            **params,
        }

        (self.output_dir / 'objective.json').write_text(json.dumps(objective, indent=2))
        log(f"[+] Objective: {objective['objective']} ({metric})")

        return objective

    def test(self, model, params):
        log("[+] Testing the model")
        test_ds = self.tune_ds(self.encode_ds(self.prepare_ds("test")))

        return self.write_results(self.labels(test_ds), self.predictions(model, test_ds), params)
