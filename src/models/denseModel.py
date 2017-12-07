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
        base_model = DenseNetImageNet121(input_shape=(75,75,3), weights=None,
                                         dropout_rate=0.2)
        x = base_model.layers[-1].get_output_at(0)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=[base_model.input],outputs=[x])
 
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


 
