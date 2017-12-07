import os
import keras
import tensorflow as tf
from keras.applications.resnet50 import identity_block, conv_block
from keras import optimizers
from keras.layers import Input, Activation, Conv2D, Dropout, Flatten, Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=True, level=modulelogger.logging.INFO)


class ResNetModel(object):
    def __init__(self, options):
        self.options = options
        self.build()
        self.model.summary()

    def build(self):
        img_input = Input(shape=(75,75,3))
        bn_axis = 3
        
        x = Conv2D(
            64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        
        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
        
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        '''
        #x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x) 
        '''
        
        x = Flatten()(x)

        x = Dense(512)(x)
        x = Activation('relu')(x)

        x = Dense(256)(x)
        x = Activation('relu')(x)

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        # Create model.
        self.model = Model(img_input, x, name='resnet50')

    def load_model(self, i):
        ckpts = [p for p in os.listdir(self.options['dir_logs']) if "model" in p]
        resume_path = os.path.join(self.options['dir_logs'], 'model_%d.h5' %i)
        logger.info("Load model from %s" % resume_path)
        self.model = keras.models.load_model(resume_path, custom_objects={'tf':tf})
        self.optimizer = self.model.optimizer
        self.loss = self.model.loss
        self.metrics = self.model.metrics
        if isinstance(self.model.layers[-2], keras.engine.training.Model):
           logger.info('Deserializing loaded model')
           self.model = self.model.layers[-2]
