from keras import backend as K
from keras.engine import Layer


class MC_Dropout(Layer):
    """Applies Monte Carlo Dropout at prediction
    """

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(MC_Dropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, _):
        return self.noise_shape

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=True)
        return inputs
