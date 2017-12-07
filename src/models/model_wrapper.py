from models.resnet import ResNetModel
from models.convnet import ConvNetModel
from models.deeperconvnet import DeeperConvNetModel
from models.denseModel import DenseNetModel

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=True, level=modulelogger.logging.INFO)

def get_wrapper(options, **kwargs):
    MODEL_NAMES = {"ResNet": ResNetModel, "ConvNet": ConvNetModel,
                   "DeeperConvNetModel":  DeeperConvNetModel, "DenseNet": DenseNetModel}
    assert options['model']['name'] in MODEL_NAMES, 'Model %s non existing' %options['model']['name']
    return MODEL_NAMES[options['model']['name']](options, **kwargs)
