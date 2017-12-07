import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Activation, Conv2D, Dropout, Flatten, Dense, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers

from models.mastermodel import MasterModel

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=True, level=modulelogger.logging.INFO)

class DeeperConvNetModel(MasterModel):
    def __init__(self, options, **kwargs):
        super(DeeperConvNetModel, self).__init__(options, **kwargs)
        self.kernel_size = options['model']['kernel_size']
        self.pool_size = options['model']['pool_size']
        self.build()
        
    def build(self):
        inputs = Input(shape=self.input_size)
       
        conv1_1 = Conv2D(64, (self.kernel_size, self.kernel_size))(inputs)
        bn1_1 = BatchNormalization()(conv1_1)
        ac1_1 = Activation('relu')(bn1_1)
        pool1 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(ac1_1)
        pool1 = Dropout(1e-2)(pool1)
        
        conv2_1 = Conv2D(128, (self.kernel_size, self.kernel_size))(pool1)
        bn2_1 = BatchNormalization()(conv2_1)
        ac2_1 = Activation('relu')(bn2_1)
        pool2 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(ac2_1)
        pool2 = Dropout(1e-2)(pool2)

        conv3_1 = Conv2D(256, (self.kernel_size, self.kernel_size))(pool2)
        bn3_1 = BatchNormalization()(conv3_1)
        ac3_1 = Activation('relu')(bn3_1)
        pool3 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(ac3_1)
        pool3 = Dropout(1e-2)(pool3)

        conv4_1 = Conv2D(512, (self.kernel_size, self.kernel_size))(pool3)
        bn4_1 = BatchNormalization()(conv4_1)
        ac4_1 = Activation('relu')(bn4_1)
        pool4 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(ac4_1)
        pool4 = Dropout(1e-2)(pool4)
        
        fl5 = Flatten()(pool4)

        dense6 = Dense(512)(fl5)
        ac6 = Activation('relu')(dense6)
        ac6 = Dropout(1e-2)(ac6)

        dense8 = Dense(1)(ac6)
        ac8 = Activation('sigmoid')(dense8)

        self.model = Model(inputs=[inputs], outputs=[ac8])
        


 
