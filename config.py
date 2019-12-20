import os.path.join as osjoin
from os.path import abspath, normpath

class Config:

    def __init__(self):

        self.openpose_weights_path = join('models', 'openpose_body25_keras.h5')
        self.openpose_config_path =