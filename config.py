import os.path.join as osjoin
from os.path import abspath, normpath


class ModelConfig:

    def __init__(self):
        self.joint_threshold = 0.1
        self.connection_threshold = 0.05
        self.tf_resize_method = 'bicubic'
        self.openpose_weights_path = osjoin('models', 'openpose_body25_keras.h5')
        self.openpose_config_path = osjoin('models', 'config')
        self.input_res = 228
        self.use_gaussian_filtering = True
        self.gaussian_kernel_sigma = 3


class HyperConfig:

    def __init__(self):
        self.use_gpu = True
        self.gpu_device_number = 0
        self.scale_search = (0.8, 1.0, 1.2)
        self.pad_value = 128
