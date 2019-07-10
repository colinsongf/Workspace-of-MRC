from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from keras import regularizers


class TrainableWeight(Layer):
    def __init__(self, size_a, size_b, reg_kernel, **kwargs):
        self.size_a = size_a
        self.size_b = size_b
        self.reg_kernel = reg_kernel
        super(TrainableWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.size_a, self.size_b),
                                      initializer=RandomNormal(mean=0.0, stddev=0.5, seed=None),
                                      regularizer=regularizers.l2(self.reg_kernel),
                                      trainable=True)
        super(TrainableWeight, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def get_config(self):
        config = {'size_a': self.size_a,
                  'size_b': self.size_b,
                  'reg_kernel': self.reg_kernel}
        base_config = super(TrainableWeight, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))