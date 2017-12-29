# -*- coding: utf-8 -*-
"""Generator used for multispectral images

This module contains the generator of batches of images and masks for segmentation 
training.

"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class CreateGenerator(object):
    def __init__(self, X_train, options):
        self.datagen = ImageDataGenerator(rotation_range=options['image_processing']['rotation_range'],
                                 vertical_flip=options['image_processing']['vertical_flip'],
                                 horizontal_flip=options['image_processing']['horizontal_flip'],
                                 fill_mode=options['image_processing']['fill_mode'],
                                 featurewise_center=True,
                                          featurewise_std_normalization=True)
        self.datagen.fit(X_train)
        
    def generator(self, X, inc, Y, batch_size=64):
        while True:
            # suffled indices    
            idx = np.random.permutation( X.shape[0])           
    
            batches = self.datagen.flow( X[idx], Y[idx], batch_size=batch_size, shuffle=False)
            idx0 = 0
            for batch in batches:
                idx1 = idx0 + batch[0].shape[0]   
                yield [batch[0], inc[ idx[ idx0:idx1 ] ]], batch[1]
    
                idx0 = idx1
                if idx1 >= X.shape[0]:
                    break
