import os
import sys
import numpy as np
import pandas as pd
from random import uniform

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=True, level=modulelogger.logging.INFO)

def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in out_df["band_1"]])
    out_images_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in out_df["band_2"]])
    out_images = np.concatenate([out_images_1[:, :, :, np.newaxis], out_images_2[:, :, :, np.newaxis],
                              ((out_images_1*out_images_2)/2)[:, :, :, np.newaxis]], axis=-1)
    return out_df, out_images
