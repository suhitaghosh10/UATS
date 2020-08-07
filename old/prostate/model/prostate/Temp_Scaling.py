import keras as k
from keras.layers import Layer


class SadLayer(Layer):

    def __init__(self, **kwargs):
        self.output_dim = (1,)
        super(SadLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.temp = self.add_weight(name='temperature',
                                    shape=(self.output_dim),
                                    initializer=k.initializers.RandomUniform(minval=1.0, maxval=1.7, seed=None),
                                    trainable=True)
        super(SadLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x / self.temp

    def compute_output_shape(self, input_shape):
        return input_shape
