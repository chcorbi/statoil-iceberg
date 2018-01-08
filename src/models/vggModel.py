import os
import keras
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, \
    concatenate, Reshape
from keras.models import Model

from models.VGG16.vgg16 import VGG16
from models.mastermodel import MasterModel

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=True, level=modulelogger.logging.INFO)

class VGGModel(MasterModel):
    def __init__(self, options, **kwargs):
        super(VGGModel, self).__init__(options, **kwargs)
        self.build()

    def build(self):
        base_model = VGG16(include_top=False, weights=None)
        x = base_model.layers[-1].get_output_at(0)

        inc_input = Input(shape=(1,))
        merged = concatenate([Reshape((-1,1))(x),Reshape((-1,1))(inc_input)], axis=1)
        fl5 = Flatten()(merged)
        
        y = Dense(4096, activation='relu', name='fc1')(fl5)
        y = Dense(4096, activation='relu', name='fc2')(y)
        y = Dense(1, activation='softmax')(y)
        self.model = Model(inputs=[base_model.input, inc_input],outputs=[y])
