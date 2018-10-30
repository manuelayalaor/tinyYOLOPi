from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os



class DenserLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(DenseLayer, self).__init()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[input_shape[-1].value,self.num_outputs])
    def call(self, input):
        return tf.matmul(input, self.kernel)

    def save_model(checkpoint_path):
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

    def load_model(checkpoint_dir):
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model = DenserLayer()
        model.load_weights(latest)
