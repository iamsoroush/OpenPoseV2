import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


class PosePredictor:

    def __init__(self, weights_path, config_path):
        self.weights_path = weights_path
        self.config_path = config_path
        self.config = self._read_config(self.config_path)

    def get_model(self):
        model = self._define_model(self.config['output_activation'], self.config['n_hidden_units'])
        model.load_weights(self.weights_path)
        return model

    @staticmethod
    def _read_config(path):
        with open(path, 'r') as f:
            config = json.load(f)
        return config

    def _define_model(self, activation, n_hidden_units):
        encoder_input = tfkl.Input(shape=(None, self.n_features), name='encoder_input')
        whole_sequence_output, final_memory_state, final_carry_state = tfkl.LSTM(n_hidden_units,
                                                                                 return_sequences=True,
                                                                                 return_state=True,
                                                                                 name='encoder_lstm')(encoder_input)
        output_tensor = tfkl.Dense(self.n_features, activation=activation, name='output_tensor')(whole_sequence_output)
        model = tfk.Model(encoder_input, output_tensor)
        return model


class PoseCorrectionPreProcessor:

    def __init__(self, kp_mapper, invis_value=-2):

        """:parameter kp_mapper: {kp_name1: 0, kp_name2: 1, ...}"""

        self.kp_mapper = kp_mapper
        self.invis_value = invis_value
        self.pose_track_kp_names = ['Nose',
                                    'head_bottom',
                                    'head_top',
                                    'LEar',
                                    'REar',
                                    'LShoulder',
                                    'RShoulder',
                                    'LElbow',
                                    'RElbow',
                                    'LWrist',
                                    'RWrist',
                                    'LHip',
                                    'RHip',
                                    'LKnee',
                                    'RKnee',
                                    'LAnkle',
                                    'RAnkle']
        self.pose_track_mapper = dict(zip(self.pose_track_kp_names, range(len(self.pose_track_kp_names))))
        self.visualizer = Visualizer(invis_value=invis_value, n_landmarks=len(self.pose_track_kp_names))

    def normalize(self, track_kps, track_bboxes):

        """Normalize given kps to (-1, 1) for using as input to posepredictor.

        :argument track_kps: np.ndarray of key-points of shape(n_kps, 2)
        :argument track_bboxes: np.ndarray([[x_min, y_min, x_max, y_max], ...])
        """

        std_kps = self._get_standard_track(track_kps, self.kp_mapper)
        isvis, isvis_ind = self._get_isvis_mat(std_kps, self.invis_value)
        x_min, y_min, x_max, y_max = self._get_lims(track_bboxes)
        normalized_track = _normalize_kps(std_kps, x_max, y_max, x_min, y_min, isvis_ind)
        flattened = normalized_track.reshape(normalized_track.shape[0], -1)
        return flattened, x_max, y_max, x_min, y_min

    def visualize_track(self, track, x_max, y_max, x_min, y_min, alpha_multiplier=0.8, ax_color='g'):

        """Visualizes track based on provided limit values.

        :argument track: np.ndarray of shape(seq_len, n_landmarks, 2) whith invisable
         landmarks filled by self.invis_value
        :argument ax: matplotlib ax object
        :argument x_max: the limit to use when drawing poses.
        :argument y_max: the limit to use when drawing poses.
        :argument x_min: the limit to use when drawing poses.
        :argument y_min: the limit to use when drawing poses.
        :argument alpha_multiplier: how much to hover poses in time.
        :argument ax_color: the graph's color
        """

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10 , 10))
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((0, y_max - y_min))
        # ax.set_title('Normalized track')

        self.visualizer.draw_sequence(track, ax, x_max, y_max, x_min, y_min, alpha_multiplier, ax_color)
        return fig, ax

    def normalize_batch(self, arr, x_max, y_max, x_min, y_min, isvis_ind=None):
        pass

    def denormalize(self, pred, x_max, y_max, x_min, y_min):

        """Denormalize the predicted keypoints.

        :argument pred: np.ndarray of shape(n_features,), predicted by pose prediction model.

        :returns res: np.ndarray of shape(n_joints, 2), which is actual values of predicted kps in original frame.
        """

        arr = np.reshape(pred, (pred.shape[0] // 2, 2))
        norm_arr = (arr + 1) / 2
        norm_arr = norm_arr * [x_max - x_min, y_max - y_min]
        norm_arr = norm_arr + [x_min, y_min]

        kps = np.zeros((len(self.kp_mapper), 2), dtype=np.float)
        kps[:] = np.nan
        for name, value in self.pose_track_mapper.items():
            if name in self.kp_mapper.keys():
                ind = self.kp_mapper[name]
                kps[ind] = norm_arr[value]
        return kps

    def _get_standard_track(self, track_kps, kp_mapper):
        std_kps = np.zeros((len(track_kps), len(self.pose_track_kp_names), 2), dtype=np.float)
        std_kps[:] = self.invis_value

        kp_names = [name for name in list(kp_mapper.keys()) if name in self.pose_track_kp_names]
        pt_indices = [self.pose_track_mapper[name] for name in kp_names]
        kp_indices = [kp_mapper[name] for name in kp_names]
        std_kps[:, pt_indices, :] = track_kps[:, kp_indices, :]

        nan_where = np.where(np.isnan(std_kps))
        if np.any(nan_where):
            std_kps[nan_where] = self.invis_value
        return std_kps

    def _get_standard_kps(self, detection_kps, kp_mapper):
        kp_names = [name for name in list(kp_mapper.keys()) if name in self.pose_track_kp_names]
        kps = np.zeros((len(self.pose_track_kp_names), 2), dtype=np.float)
        kps[:] = self.invis_value
        for kp_name in kp_names:
            kp = detection_kps[kp_mapper[kp_name]]
            if np.any(np.isnan(kp)):
                continue
            else:
                kps[self.pose_track_mapper[kp_name]] = kp

        return kps

    @staticmethod
    def _get_isvis_mat(arr, invis_value):
        isvis = np.ones((arr.shape[:-1]))
        ind1, ind2, _ = np.where(arr == invis_value)
        isvis[ind1, ind2] = 0
        return isvis, (ind1, ind2)

    @staticmethod
    def _get_lims(bboxes):
        """Returns limits of track window.

        :argument bboxes: np.ndarray([[x_min, y_min, x_max, y_max], ...])
        """

        x_min, y_min, _, _ = np.min(bboxes, axis=0)
        _, _, x_max, y_max = np.max(bboxes, axis=0)

        return x_min, y_min, x_max, y_max


class Visualizer:

    def __init__(self, invis_value=-2, n_landmarks=17):
        self.n_landmarks = n_landmarks
        self.invis_value = invis_value
        self.sks = [[15, 13],
                    [13, 11],
                    [16, 14],
                    [14, 12],
                    [11, 12],
                    [5, 11],
                    [6, 12],
                    [5, 6],
                    [5, 7],
                    [6, 8],
                    [7, 9],
                    [8, 10],
                    [1, 2],
                    [0, 1],
                    [0, 2],
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6]]

    def draw_sequence(self, seq, ax, x_max, y_max, x_min, y_min, alpha_multiplier=0.8, ax_color='g'):
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((0, y_max - y_min))

        j = 1
        for i in range(len(seq)):
            sample = seq[i].copy()

            vis = self._get_vis_vector(sample)

            sample[:, 1] = y_max - sample[:, 1]

            self._plot_pose(sample, vis, self.sks, ax_color, j, ax)

            j = j * alpha_multiplier

    def _get_vis_vector(self, kps):
        x = kps[:, 0]
        y = kps[:, 1]

        v = np.ones(self.n_landmarks)
        v[3: 5] = 0
        a = np.where(x == self.invis_value)[0].tolist()
        b = np.where(y == self.invis_value)[0].tolist()
        zero_out = np.intersect1d(a, b)
        if np.any(zero_out):
            v[zero_out] = 0
        return v

    @staticmethod
    def _prepare_sample_for_vis(sample, invis_val):
        prepared_sample = np.zeros((sample.shape[0], 3, sample.shape[1] // 2))
        for i, kp in enumerate(sample):
            x = kp[0::2]
            y = kp[1::2]

            v = np.ones(len(x))
            a = np.where(x == invis_val)[0].tolist()
            b = np.where(y == invis_val)[0].tolist()
            zero_out = np.intersect1d(a, b)
            if np.any(zero_out):
                v[zero_out] = 0
            v[3: 5] = 0
            prepared_sample[i] = [x, y, v]
        return prepared_sample

    @staticmethod
    def _plot_pose(kp, v, sks, c, alpha, ax):
        # x = kp[0]
        # y = kp[1]
        #
        # for sk in sks:
        #     if np.all(v[sk] > 0):
        #         ax.plot(x[sk], y[sk], linewidth=3, color=c, alpha=alpha)
        # ax.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k', markeredgewidth=2,
        #         alpha=alpha)

        x = kp[:, 0]
        y = kp[:, 1]

        for sk in sks:
            if np.all(v[sk] > 0):
                ax.plot(x[sk], y[sk], linewidth=3, color=c, alpha=alpha)
        ax.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k', markeredgewidth=2,
                alpha=alpha)


def load_pose_predictor(model_path, config_path):
    model = tfk.models.load_model(model_path)
    if config_path:
        config = read_json(config_path)
    else:
        config = None

    return model, config


def read_json(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def _normalize_kps(arr, x_max, y_max, x_min, y_min, isvis_ind=None):
    """returns normalized(-1, 1) array."""

    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3
    assert x_max != x_min
    assert y_max != y_min

    shape = arr.shape
    len_seq, n_landmarks = shape[:2]

    to_sub = np.tile([x_min, y_min], (len_seq, n_landmarks, 1))
    if isvis_ind is not None:
        to_sub[isvis_ind] = [0, 0]

    to_mul = np.tile([2 / (x_max - x_min), 2 / (y_max - y_min)], (len_seq, n_landmarks, 1))
    if isvis_ind is not None:
        to_mul[isvis_ind] = [1, 1]

    sub_one = -1 * np.ones(shape)
    if isvis_ind is not None:
        sub_one[isvis_ind] = [0, 0]

    normalized = (to_mul * (arr - to_sub)) + sub_one
    return normalized
