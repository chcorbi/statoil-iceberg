import os
import argparse
import yaml
import numpy as np
import pandas as pd
import e3_tools.fileSystem.utils as utils

import callback
from resnet import ResNetModel
from convnet import ConvNetModel
from denseModel import DenseNetModel
from dataset import load_and_format
from parallelizer import Parallelizer

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=True, level=modulelogger.logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser() # Parsing of user arguments
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--parallelize', action='store_true', help="Parallelize model in GPU")
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] 
        logger.info('Working on GPUs #%s' %str(gpu_list))
    else:
        gpu_list = []

    with open(os.path.join(args.config_path,'args.yaml'), 'r') as handle:
        options = yaml.load(handle)
        
    logger.info('Loading test dataset')
    test_df, test_images = load_and_format(os.path.join(options['dataset']['path'], 'test.json'))
    print('test', test_df.shape, 'loaded', test_images.shape)

    y_pred = np.zeros(test_images.shape[0])
    
    for i in range(5):
        logger.info('Predict %d/5' %(i+1))
        model_wrapper = ConvNetModel(options)
        #model_wrapper = ResNetModel(options)
        #model_wrapper = DenseNetModel(options)
        model_wrapper.load_model(i)
        
        if args.parallelize:
            logger.info('Parallelizing model on GPUs')
            if len(gpu_list)==1:
                logger.warning('Cannot parallelize, available GPUS < 2')
            else:
                parallelizer = Parallelizer(gpu_list=range(0,len(gpu_list)))
                model_wrapper.model = parallelizer.transform(model_wrapper.model)
                
        
        
        logger.info('Predict test set')
        y_pred += model_wrapper.model.predict(test_images).reshape((y_pred.shape[0]))

    y_pred /= 5
    
    logger.info('Write csv file')
    submission = pd.DataFrame()
    submission['id']=test_df['id']
    submission['is_iceberg']= y_pred.reshape((y_pred.shape[0]))
    submission.to_csv(os.path.join(args.config_path, 'submission.csv'), index=False, float_format='%.6f')
