import os
import argparse
import yaml
import numpy as np
import pandas as pd
import e3_tools.fileSystem.utils as utils
from keras.preprocessing.image import ImageDataGenerator

from lib import dataset
from lib import callback
from lib.gen import CreateGenerator
from lib.parallelizer import Parallelizer
from models import model_wrapper as modelw

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
    test_df, test_images = dataset.load_and_format(os.path.join(options['dataset']['path'], 'test.json'))
    inc_test = np.array(test_df['inc_angle'])
    print('test', test_df.shape, 'loaded', test_images.shape)

    train_df, train_images = dataset.load_and_format(os.path.join(options['dataset']['path'], 'train.json'))
    print('training', train_df.shape, 'loaded', train_images.shape)
    datagen = ImageDataGenerator(rotation_range=options['image_processing']['rotation_range'],
                                 vertical_flip=options['image_processing']['vertical_flip'],
                                 horizontal_flip=options['image_processing']['horizontal_flip'],
                                 fill_mode=options['image_processing']['fill_mode'],
                                 featurewise_center=True,
                                 featurewise_std_normalization=True)
    
    logger.info('Fitting generator')
    creategen = CreateGenerator(train_images, options)

    logger.info('Standardizing test images')
    y_pred = np.zeros(test_images.shape[0])
    for j in range(len(test_images)):
        test_images[j] = creategen.datagen.standardize(test_images[j])
        
    for i in range(options['model']['stacking']):
        logger.info('Predict %d/%d' %((i+1), options['model']['stacking']))
        model_wrapper = modelw.get_wrapper(options)
        model_wrapper.load_model(i)
        
        if args.parallelize:
            logger.info('Parallelizing model on GPUs')
            if len(gpu_list)==1:
                logger.warning('Cannot parallelize, available GPUS < 2')
            else:
                parallelizer = Parallelizer(gpu_list=range(0,len(gpu_list)))
                model_wrapper.model = parallelizer.transform(model_wrapper.model)
        
        logger.info('Predict test set')
        y_pred += model_wrapper.model.predict([test_images,inc_test]).reshape((y_pred.shape[0]))

    y_pred /= options['model']['stacking']
    
    logger.info('Write csv file')
    submission = pd.DataFrame()
    submission['id']=test_df['id']
    submission['is_iceberg']= y_pred.reshape((y_pred.shape[0]))
    submission.to_csv(os.path.join(args.config_path, 'submission.csv'), index=False, float_format='%.6f')
