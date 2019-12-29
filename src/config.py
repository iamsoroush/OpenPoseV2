from os.path import normpath, abspath, join, dirname, realpath
import numpy as np


class OpenPoseV2Config:

    def __init__(self):
        self.joint_threshold = 0.1
        self.connection_threshold = 0.05
        self.resize_method = 'bicubic'
        project_root = normpath(abspath(dirname(dirname(realpath(__file__)))))
        self.weights_path = join(project_root, 'models', 'openpose_v2', 'openpose_body25_keras.h5')
        self.input_res = 228
        self.use_gaussian_filtering = True
        self.gaussian_kernel_sigma = 3


class HyperConfig:

    def __init__(self):
        self.use_gpu = True
        self.gpu_device_number = 0
        self.scale_search = (0.8, 1.0, 1.2)
        self.pad_value = 128
        self.drawing_stick = 10
        self.kp_names = ["Nose",
                         "Neck",
                         "RShoulder",
                         "RElbow",
                         "RWrist",
                         "LShoulder",
                         "LElbow",
                         "LWrist",
                         "MidHip",
                         "RHip",
                         "RKnee",
                         "RAnkle",
                         "LHip",
                         "LKnee",
                         "LAnkle",
                         "REye",
                         "LEye",
                         "REar",
                         "LEar",
                         "LBigToe",
                         "LSmallToe",
                         "LHeel",
                         "RBigToe",
                         "RSmallToe",
                         "RHeel",
                         "Background"]
        self.kp_mapper = dict(zip(range(len(self.kp_names)), self.kp_names))
        self.connections = [[1, 8],
                            [1, 2],
                            [1, 5],
                            [2, 3],
                            [3, 4],
                            [5, 6],
                            [6, 7],
                            [8, 9],
                            [9, 10],
                            [10, 11],
                            [8, 12],
                            [12, 13],
                            [13, 14],
                            [1, 0],
                            [0, 15],
                            [15, 17],
                            [0, 16],
                            [16, 18],
                            # [2, 17],
                            # [5, 18],
                            [14, 19],
                            [19, 20],
                            [14, 21],
                            [11, 22],
                            [22, 23],
                            [11, 24]]
        self.map_paf_to_connections = [[0, 1],
                                       [14, 15],
                                       [22, 23],
                                       [16, 17],
                                       [18, 19],
                                       [24, 25],
                                       [26, 27],
                                       [6, 7],
                                       [2, 3],
                                       [4, 5],
                                       [8, 9],
                                       [10, 11],
                                       [12, 13],
                                       [30, 31],
                                       [32, 33],
                                       [36, 37],
                                       [34, 35],
                                       [38, 39],
                                       # [20, 21],
                                       # [28, 29],
                                       [40, 41],
                                       [42, 43],
                                       [44, 45],
                                       [46, 47],
                                       [48, 49],
                                       [50, 51]]

        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 255, 0], [170, 255, 0], [85, 255, 0],
                       [0, 255, 0], [0, 255, 85], [0, 255, 170],
                       [0, 255, 255], [0, 170, 255], [0, 85, 255],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [255, 0, 255], [255, 0, 170], [255, 0, 85],
                       [255, 170, 85], [255, 170, 170], [255, 170, 255],
                       [255, 85, 85], [255, 85, 170], [255, 85, 255],
                       [170, 170, 170]]


class FeatureExtractorConfig:

    def __init__(self):
        self.points_comb = np.array([[4, 3, 2],
                                    [3, 2, 1],
                                    [1, 5, 6],
                                    [5, 6, 7],
                                    [2, 1, 0],
                                    [2, 1, 8],
                                    [1, 8, 9],
                                    [1, 8, 12],
                                    [8, 9, 10],
                                    [9, 10, 11],
                                    [10, 11, 22],
                                    [8, 12, 13],
                                    [12, 13, 14],
                                    [13, 14, 19]])
