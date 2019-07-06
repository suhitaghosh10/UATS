import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer


class MyLayer(Layer):

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._temp = K.variable(1.5)
        self.trainable_weights = [self._temp]

        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inp):
        return tf.divide(inp, self._temp)

    def compute_output_shape(self, input_shape):
        return input_shape[0]
