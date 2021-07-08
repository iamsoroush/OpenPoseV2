from os.path import join

from . import PROJECT_ROOT

KP_NAMES = ["Nose",
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
MODEL_NAME = 'openpose_body25_keras.h5'


class OpenPoseV2Config:

    # JOINT_THRESHOLD = 0.1
    # CONNECTION_THRESHOLD = 0.05
    # MIN_VISIBLE_PARTS = 4
    # RESIZE_METHOD = 'bicubic'
    # WEIGHTS_PATH = join(PROJECT_ROOT, 'models', 'openpose_v2', 'openpose_body25_keras.h5')
    # INPUT_RES = 368
    # USE_GAUSSIAN_FILTERING = True
    # GAUSSIAN_KERNEL_SIGMA = 3

    def __init__(self):
        self.joint_threshold = 0.1
        self.connection_threshold = 0.05
        self.min_vis_parts = 4
        self.resize_method = 'bicubic'
        self.weights_path = join(PROJECT_ROOT, 'model', MODEL_NAME)
        self.input_res = 368
        self.use_gaussian_filtering = True
        self.gaussian_kernel_sigma = 3


class HyperConfig:

    def __init__(self):
        self.use_gpu = True
        self.gpu_device_number = 0
        self.pad_value = 128
        self.drawing_stick = 5
        self.scales = (0.8, 1.0, 1.2)

        self.kp_mapper = dict(zip(range(len(KP_NAMES)), KP_NAMES))
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
