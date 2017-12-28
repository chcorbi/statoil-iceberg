import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Flatten, Reshape
from models.DenseNet.densenet import DenseNetImageNet121

from models.mastermodel import MasterModel

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=True, level=modulelogger.logging.INFO)

class DenseNetModel(MasterModel):
    def __init__(self, options, **kwargs):
        super(DenseNetModel, self).__init__(options, **kwargs)
        self.kernel_size = options['model']['kernel_size']
        self.pool_size = options['model']['pool_size']
        self.build()

    def build(self):
        base_model = DenseNetImageNet121(input_shape= self.input_size, weights='imagenet',
                                         dropout_rate=0.1)
        x = base_model.layers[-1].get_output_at(0)

        inc_input = Input(shape=(1,))
        merged = concatenate([Reshape((-1,1))(x),Reshape((-1,1))(inc_input)], axis=1)
        fl5 = Flatten()(merged)
        
        y = Dense(1, activation='sigmoid')(fl5)
        self.model = Model(inputs=[base_model.input, inc_input],outputs=[y])
 



 
