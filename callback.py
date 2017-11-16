import os
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,CSVLogger,TensorBoard,LambdaCallback

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=False, level=modulelogger.logging.INFO)

       
def define_callbacks(ckpt_rootpath):
    ckpt_callback = ModelCheckpoint(os.path.join(ckpt_rootpath,'model.h5'),
                                    monitor='val_loss',
                                    save_best_only=True,
                                    mode='auto',save_weights_only=False) # Saves the model weights after each epoch
    earlystopping_callback = EarlyStopping(monitor='val_loss',
                                           patience=20,
                                           verbose=1) # Early stopping
    reducelr_callback = ReduceLROnPlateau(patience=5,
                                          verbose=1) # LR decay 
    Tb = TensorBoard(log_dir=ckpt_rootpath)
    callbacks = [#reducelr_callback,
                 ckpt_callback,
                 earlystopping_callback,
                 Tb]
    return callbacks
