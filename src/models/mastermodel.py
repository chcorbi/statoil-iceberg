import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import optimizers
from math import sqrt
from osgeo import gdal

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=True, level=modulelogger.logging.INFO)

MAX_IMG_BF_SIZE = 16

def rescaling_func(x, bit_size):
   return (x / float(2**bit_size - 1)) - 0.5

class MasterModel(object):
    def __init__(self, options, **kwargs):
        self.options = options
        self.input_size = (options['dataset']['img_size'],options['dataset']['img_size'])
        self.nbands = options['dataset']['img_nbands']
        self.lr = self.options['optim']['init_lr']

    def build(self):
        raise NotImplemented

    def set_optimizer(self):
      self.loss = self.options['optim']['loss']
      self.metrics = self.options['optim']['metrics']
      OPTIMIZERS = {'Adam': optimizers.Adam(lr=self.lr),
                    'SGD': optimizers.SGD(lr=self.lr)}
      assert self.options['optim']['optimizer'] in OPTIMIZERS, 'Optimizer `%s` non implemented' %self.options['optim']['optimizer']
      self.optimizer =  OPTIMIZERS[self.options['optim']['optimizer']]

    def load_from_weights(self, resume_path):
        model_path = resume_path[:-11]+".h5"
        if not os.path.isfile(resume_path):
            if os.path.isfile(model_path):
                model = keras.models.load_model(model_path, custom_objects={'tf':tf})
                if isinstance(model.layers[-2], keras.engine.training.Model):
                  logger.info('Deserializing loaded model')
                  model = model.layers[-2]
                #model.save_weights(resume_path)
                self.model = model
                return
        try:
            self.model.load_weights(resume_path)
            logger.info("Loading weights file model from %s" % resume_path)
        except IOError as e:
            logger.info("No weights file for current model. Will be created.")
        except:
            raise


    def load_model(self, model_path):
        logger.info("Load model from %s" % model_path)
        self.model = keras.models.load_model(model_path, custom_objects={'tf':tf})
        self.optimizer = self.model.optimizer
        self.loss = self.model.loss
        self.metrics = self.model.metrics
        if isinstance(self.model.layers[-2], keras.engine.training.Model):
           logger.info('Deserializing loaded model')
           self.model = self.model.layers[-2]

    def compile(self):
        logger.info('Compiling model')
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)
