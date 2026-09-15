import tensorflow as tf

from config import COMPILER_LABELS
from experiment.experiment import Experiment


class CompilerExperiment(Experiment):
    def num_classes(self):
        return len(COMPILER_LABELS)

    def validation_metric(self):
        return "val_binary_accuracy"

    def get_output_layers(self):
        return [tf.keras.layers.LSTM(256, activation='tanh',
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.5 * 0.125)),
                tf.keras.layers.Dense(self.num_classes(), activation='softmax')]

    def compile_model(self, model):
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      optimizer=tf.keras.optimizers.Adam(self.initial_learning_rate),
                      metrics=[tf.keras.metrics.BinaryAccuracy()])
