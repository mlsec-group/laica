import tensorflow as tf

from config import MAX_NUM_PARAMS
from experiment.experiment import Experiment


class NParamExperiment(Experiment):
    def num_classes(self):
        return MAX_NUM_PARAMS

    def get_output_layers(self):
        return [tf.keras.layers.GRU(256, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.GRU(256, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.GRU(256, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.num_classes())]

    def compile_model(self, model):
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(self.initial_learning_rate),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
