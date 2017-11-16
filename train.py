import os
import argparse
import click
import yaml
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import callback
from resnet import ResNetModel
from convnet import ConvNetModel
from dataset import load_and_format
from parallelizer import Parallelizer

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
    train_df, train_images = load_and_format(os.path.join(options['dataset']['path'], 'train.json'))
    print('training', train_df.shape, 'loaded', train_images.shape)

    logger.info('Splitting train and val images')
    X_train, X_valid, y_train, y_valid = train_test_split(train_images,
                                                   train_df['is_iceberg'],
                                                    random_state = 2017,
                                                    test_size = 0.20
                                                   )
    print('Train', X_train.shape, y_train.shape)
    print('Validation', X_valid.shape, y_valid.shape)

    logger.info('Build model')
    model_wrapper = ConvNetModel(options)
    
    if args.parallelize:
        logger.info('Parallelizing model on %d GPUs' %len(gpu_list))
        if len(gpu_list)==1:
            logger.warning('Cannot parallelize, available GPUS < 2')
        else:
            parallelizer = Parallelizer(gpu_list=range(0,len(gpu_list)))
            model_wrapper.model = parallelizer.transform(model_wrapper.model)

    model_wrapper.model.compile(loss=options['optim']['loss'],
            optimizer=options['optim']['optimizer'],
            metrics=options['optim']['metrics'])

    callbacks = callback.define_callbacks(options['dir_logs'])

    logger.info('Start fitting model')
    model_wrapper.model.fit(X_train, y_train,
          batch_size=options['optim']['batch_size'],
          epochs=options['optim']['epochs'],
          verbose=1,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)

    logger.info('Validation score')
    model_wrapper.load_model()
    score = model_wrapper.model.evaluate(X_valid, y_valid, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])