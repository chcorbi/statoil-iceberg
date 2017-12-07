import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense
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
        base_model = DenseNetImageNet121(input_shape= self.input_size, weights=None,
                                         dropout_rate=0.)
        x = base_model.layers[-1].get_output_at(0)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=[base_model.input],outputs=[x])
 



 
