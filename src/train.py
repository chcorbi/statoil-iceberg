import os
import argparse
import click
import yaml
import sys
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.utils.np_utils import to_categorical

from lib import dataset
from lib import callback
from lib.gen import CreateGenerator
from lib.parallelizer import Parallelizer
from models import model_wrapper as modelw

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=True, level=modulelogger.logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="options/default.yaml", help='Yaml configuration file location.')
    parser.add_argument('--dir_logs', type=str, default="", help='Logs directory')
    parser.add_argument('--weights', default=None, help="Load existing weights file")
    parser.add_argument('--parallelize', action='store_true', help="Parallelize model in GPU")
    parser.add_argument('--resume', action='store_true', help="Load model from a previous experiment")
    parser.add_argument('--continuing', action='store_true', help="Continue training")
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES']
        logger.info('Working on GPUs #%s' %gpu_list)
        gpu_list = gpu_list.split(',')

    else:
        gpu_list = []
        logger.warning('No GPUs specified, working by default on all available.')

    with open(args.config_path, 'r') as handle:
        options = yaml.load(handle)

    if args.dir_logs != '':
        options['dir_logs'] = args.dir_logs

    if args.continuing:
        args.resume = True

    if args.resume:
        assert os.path.isdir(options['dir_logs']), "Tried to resume a non existing model"
    else:
        if os.path.isdir(options['dir_logs']):
            if click.confirm('Logs directory already exists in {}. Erase?'
                    .format(options['dir_logs'], default=False)):
                os.system('rm -r ' + options['dir_logs'])
            else:
                raise
        os.makedirs(options['dir_logs'])
        with open(os.path.join(options['dir_logs'],'args.yaml'), 'w') as f:
            yaml.dump(options, f, default_flow_style=False)

    logger.info('Loading training dataset')
    train_df, train_images = dataset.load_and_format(os.path.join(options['dataset']['path'], 'train.json'))
    print('training', train_df.shape, 'loaded', train_images.shape)

    if os.path.exists(os.path.join(options['dataset']['path'], "class_weight.txt")):
        logger.info('Loading class weights')
        with open(os.path.join(options['dataset']['path'], "class_weight.txt"), 'r') as f:
            class_weight = [float(line.rstrip('\n')) for line in f]
    else:    
        class_weight = dataset.compute_class_weight(options['dataset']['path'], train_df)
   
    sss = StratifiedShuffleSplit(n_splits=options['model']['stacking'], test_size=0.2, random_state=42)

    for i,(train_index,test_index) in enumerate(sss.split(train_images, train_df['is_iceberg'])):
        logger.info ('Training %d/%d' %((i+1), options['model']['stacking']))
        logger.info('Splitting train and val images')
        X_train, X_valid = train_images[train_index], train_images[test_index]
        inc_train, inc_valid = np.array(train_df['inc_angle'][train_index]), np.array(train_df['inc_angle'][test_index])
        y_train, y_valid =  np.array(train_df['is_iceberg'][train_index]),  np.array(train_df['is_iceberg'][test_index])
        print('Train', X_train.shape, y_train.shape)
        print('Validation', X_valid.shape, y_valid.shape)

        logger.info('Fitting generator')
        creategen = CreateGenerator(X_train, options)
        
        logger.info('Initialize generator')
        train_gen = creategen.generator(X_train, inc_train, y_train, batch_size=options['optim']['batch_size'])

        for j in range(len(X_valid)):
            X_valid[j] = creategen.datagen.standardize(X_valid[j])
            
        logger.info('Build model')
        model_wrapper = modelw.get_wrapper(options)

        if args.resume:
            logger.info('Resume model')
            model_wrapper.load_model(i)
        
        if args.parallelize:
            logger.info('Parallelizing model on %d GPUs' %len(gpu_list))
            if len(gpu_list)==1:
                logger.warning('Cannot parallelize, available GPUS < 2')
            else:
                parallelizer = Parallelizer(gpu_list=range(0,len(gpu_list)))
                model_wrapper.model = parallelizer.transform(model_wrapper.model)

        
        if args.continuing:
            logger.warning('Continue training with saved state')
        else:
            if args.resume:
                logger.warning('Setting new state for loaded model')
            model_wrapper.set_optimizer()

        model_wrapper.compile()

        callbacks = callback.define_callbacks(options['dir_logs'], i)
                
        # fits the model on batches with real-time data augmentation:
        logger.info('Start fitting model')
        model_wrapper.model.fit_generator(train_gen,
                        steps_per_epoch= len(X_train) // options['optim']['batch_size'],
                                          epochs=options['optim']['epochs'], class_weight=class_weight,
                                          validation_data=([X_valid, inc_valid], y_valid), callbacks=callbacks)

        logger.info('Validation score')

        score = model_wrapper.model.evaluate([X_valid,inc_valid], y_valid, verbose=1)
        print('Val loss:', score[0])
        print('Val accuracy:', score[1])

