import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Activation, Conv2D, Dropout, Flatten, Dense, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=True, level=modulelogger.logging.INFO)

class ConvNetModel(object):
    def __init__(self, options):
        self.options = options
        self.kernel_size = options['model']['kernel_size']
        self.pool_size = options['model']['pool_size']
        self.build()
        
    def build(self):
        inputs = Input(shape=(75,75,3))
       
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

        conv3_1 = Conv2D(128, (self.kernel_size, self.kernel_size))(pool2)
        bn3_1 = BatchNormalization()(conv3_1)
        ac3_1 = Activation('relu')(bn3_1)
        pool3 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(ac3_1)
        pool3 = Dropout(1e-2)(pool3)

        conv4_1 = Conv2D(64, (self.kernel_size, self.kernel_size))(pool3)
        bn4_1 = BatchNormalization()(conv4_1)
        ac4_1 = Activation('relu')(bn4_1)
        pool4 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(ac4_1)
        pool4 = Dropout(1e-2)(pool4)
        
        fl5 = Flatten()(pool4)

        dense6 = Dense(512)(fl5)
        ac6 = Activation('relu')(dense6)
        ac6 = Dropout(1e-2)(ac6)

        dense7 = Dense(256)(ac6)
        ac7 = Activation('relu')(dense7)
        ac7 = Dropout(1e-2)(ac7)

        dense8 = Dense(1)(ac7)
        ac8 = Activation('sigmoid')(dense8)

        self.model = Model(inputs=[inputs], outputs=[ac8])

    def load_model(self):
        ckpts = [p for p in os.listdir(self.options['dir_logs']) if "model" in p]
        resume_path = os.path.join(self.options['dir_logs'], 'model.h5')
        logger.info("Load model from %s" % resume_path)
        self.model = keras.models.load_model(resume_path, custom_objects={'tf':tf})
        self.optimizer = self.model.optimizer
        self.loss = self.model.loss
        self.metrics = self.model.metrics
        if isinstance(self.model.layers[-2], keras.engine.training.Model):
           logger.info('Deserializing loaded model')
           self.model = self.model.layers[-2]


 
