from configobj import ConfigObj
from time import time

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfb
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow.keras.backend as tfkb

import cv2


class OpenPoseV2:

    kp_names = ["Nose",
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
    kp_mapper = dict(zip(range(len(kp_names)), kp_names))
    connections = [[1, 8],
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
    map_paf_to_connections = [[0, 1],
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

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
              [255, 255, 0], [170, 255, 0], [85, 255, 0],
              [0, 255, 0], [0, 255, 85], [0, 255, 170],
              [0, 255, 255], [0, 170, 255], [0, 85, 255],
              [0, 0, 255], [85, 0, 255], [170, 0, 255],
              [255, 0, 255], [255, 0, 170], [255, 0, 85],
              [255, 170, 85], [255, 170, 170], [255, 170, 255],
              [255, 85, 85], [255, 85, 170], [255, 85, 255],
              [170, 170, 170]]

    def __init__(self,
                 weights_path,
                 config_path,
                 input_shape=(512, 512),
                 gaussian_filtering=True):
        self.openpose_model = FastOpenPoseV2Model(weights_path,
                                                  config_path,
                                                  input_shape,
                                                  gaussian_filtering)
        self.model = self.openpose_model.load_model()
        # self.fe = FeatureExtractor()
        self.n_joints = len(OpenPoseV2.kp_mapper) - 1
        self.n_limbs = len(OpenPoseV2.connections)
        self.drawing_stick = 15

    def draw_pose(self, img):
        org_h, org_w, _ = img.shape

        resized = tf.image.resize_with_pad(img, self.openpose_model.input_h, self.openpose_model.input_w).numpy()
        t = time()
        peaks, subset, candidate = self._inference(resized)
        print('complete inference: ', time() - t)

        if not subset.any():
            return img
        else:
            transformed_candidate = self.inverse_transform_kps(org_h, org_w,
                                                               self.openpose_model.input_h,
                                                               self.openpose_model.input_w,
                                                               candidate)

        drawed = img.copy()

        for i, person in enumerate(subset):
            t = time()
            kps = self._extract_keypoints(person, transformed_candidate)
            print('KP extraction: ', time() - t)

            self._draw_kps(drawed, kps)
            drawed = self._draw_connections(drawed, person, transformed_candidate)
        return drawed

    def _draw_kps(self, img, kps):
        for i, kp in enumerate(kps):
            if kp is not None:
                cv2.circle(img, (int(kp[0]), int(kp[1])), self.drawing_stick, OpenPoseV2.colors[i], thickness=-1)

    def _draw_connections(self, img, person, transformed_candidate):
        for i in range(self.n_limbs):
            index = person[np.array(OpenPoseV2.connections[i])]
            if -1 in index:
                continue
            cur_canvas = img.copy()
            y = transformed_candidate[index.astype(int), 0]
            x = transformed_candidate[index.astype(int), 1]
            m_x = np.mean(x)
            m_y = np.mean(y)
            length = np.sqrt(np.power(x[0] - x[1], 2) + np.power(y[0] - y[1], 2))
            angle = np.degrees(np.arctan2(x[0] - x[1], y[0] - y[1]))
            polygon = cv2.ellipse2Poly((int(m_y), int(m_x)),
                                       (int(length / 2), self.drawing_stick),
                                       int(angle),
                                       0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, OpenPoseV2.colors[i])
            img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
        return img

    @staticmethod
    def inverse_transform_kps(org_h, org_w, h, w, candidate):
        kps = candidate[:, 0: 2].astype(np.int)
        scale_factor = np.max([org_h, org_w]) / h
        transformed_candidate = np.zeros((candidate.shape[0], 3))
        if org_h > org_w:
            resized_w = org_w / scale_factor
            border = (w - resized_w) / 2
            for i, kp in enumerate(kps):
                transformed_candidate[i, 0] = scale_factor * (kp[0] - border)
                transformed_candidate[i, 1] = scale_factor * kp[1]
                transformed_candidate[i, 2] = candidate[i, 2]
        else:
            resized_h = org_h / scale_factor
            border = (h - resized_h) / 2
            for i, kp in enumerate(kps):
                transformed_candidate[i, 0] = scale_factor * kp[0]
                transformed_candidate[i, 1] = scale_factor * (kp[1] - border)
                transformed_candidate[i, 2] = candidate[i, 2]
        return transformed_candidate

    def _extract_keypoints(self, person_subset, candidate_arr):
        kps = list()
        for i in range(self.n_joints):
            kp_ind = person_subset[i].astype(np.int)
            if kp_ind == -1:
                kps.append(None)
            else:
                kps.append(candidate_arr[kp_ind, 0: 2].astype(np.int))
        return kps

    def _inference(self, img):
        """Img must be of shape (self.box_size // 2, self.box_size // 2), i.e. (184, 184)."""

        paf, masked_heatmap = self.model.predict(np.expand_dims(img, axis=0))
        all_peaks = self._get_peaks(masked_heatmap)
        connection_all, special_k = self._get_connections(paf[0], all_peaks)
        subset, candidate = self._get_subset(all_peaks, special_k, connection_all)
        return all_peaks, subset, candidate

    def _get_peaks(self, masked_heatmap):
        t = time()
        ys, xs, channels = np.nonzero(masked_heatmap[0])

        all_peaks = list()
        peak_counter = 0
        for i in range(self.n_joints):
            indices = np.where(channels == i)[0]
            n_peaks = len(indices)
            peak_inds = range(peak_counter, peak_counter + n_peaks)
            part_peaks = list()
            for j, ind in enumerate(indices):
                x = xs[ind]
                y = ys[ind]
                conf_score = masked_heatmap[0, y, x, i]
                part_peaks.append((x, y, conf_score, peak_inds[j]))
            all_peaks.append(part_peaks)
            peak_counter = peak_counter + n_peaks
        print('_get_peaks: ', time() - t)
        return all_peaks

    def _get_connections(self, paf, all_peaks):
        t = time()
        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(OpenPoseV2.map_paf_to_connections)):
            score_mid = paf[:, :, OpenPoseV2.map_paf_to_connections[k]]
            cand_a = all_peaks[OpenPoseV2.connections[k][0]]
            cand_b = all_peaks[OpenPoseV2.connections[k][1]]
            n_a = len(cand_a)
            n_b = len(cand_b)
            if n_a != 0 and n_b != 0:
                connection_candidate = []
                for i in range(n_a):
                    for j in range(n_b):
                        vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                        norm = np.linalg.norm(vec)
                        # failure case when 2 body parts overlaps
                        if norm == 0:
                            continue
                        vec = np.divide(vec, norm)

                        points_y_pos = np.round(np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num)).astype(np.int)
                        points_x_pos = np.round(np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)).astype(np.int)

                        # mid_points_vector = score_mid[points_x_pos, points_y_pos]
                        # mid_points_score = np.sum(mid_points_vector * vec, axis=0)
                        # score_with_dist_prior = mid_points_score.mean() + min(self.openpose_model.input_h / 2 / norm, 0)
                        #
                        # criterion1 = len(np.nonzero(mid_points_score > self.openpose_model.thre2)[0]) > 0.8 * len(
                        #     mid_points_score)
                        # criterion2 = score_with_dist_prior > 0
                        # if criterion1 and criterion2:
                        #     connection_candidate.append([i,
                        #                                  j,
                        #                                  score_with_dist_prior,
                        #                                  score_with_dist_prior + cand_a[i][2] + cand_b[j][2]])

                        vec_x = score_mid[points_x_pos, points_y_pos, 0]
                        vec_y = score_mid[points_x_pos, points_y_pos, 1]

                        score_mid_pts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_mid_pts) / len(score_mid_pts) + min(
                            0.5 * self.openpose_model.input_h / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_mid_pts > self.openpose_model.thre2)[0]) > 0.8 * len(
                            score_mid_pts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior,
                                                         score_with_dist_prior + cand_a[i][2] + cand_b[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                        if len(connection) >= min(n_a, n_b):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])
        print('_get_connections: ', time() - t)
        return connection_all, special_k

    def _get_subset(self,
                    all_peaks,
                    special_k,
                    connection_all):
        t = time()
        subset = -1 * np.ones((0, self.n_joints + 3))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(OpenPoseV2.map_paf_to_connections)):
            if k not in special_k:
                part_as = connection_all[k][:, 0]
                part_bs = connection_all[k][:, 1]
                index_a, index_b = np.array(OpenPoseV2.connections[k])

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][index_b] != part_bs[i]:
                            subset[j][index_b] = part_bs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][index_b] = part_bs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < self.n_limbs:
                        row = -1 * np.ones(self.n_joints + 3)
                        row[index_a] = part_as[i]
                        row[index_b] = part_bs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete some rows of subset which has few parts occur
        delete_idx = []
        for i in range(len(subset)):
            if subset[i][-1] < 8 or subset[i][-2] / subset[i][-1] < 0.4:
                delete_idx.append(i)
        subset = np.delete(subset, delete_idx, axis=0)
        print('_get_subset: ', time() - t)
        return subset, candidate


class FastOpenPoseV2Model:

    """Fast OpenPose body25 model.

    Key-point extraction implemented as graph definitions

    Note: pass the image in shape=input_shape as RGB(0, 255)
    """

    def __init__(self,
                 weights_path,
                 config_path,
                 input_shape,
                 gaussian_filtering):
        self.weights_path = weights_path
        self.config_path = config_path
        self.params, self.model_params = self._read_config()
        self.stride = self.model_params['stride']
        self.pad_value = self.model_params['padValue']
        self.box_size = self.model_params['boxsize']
        self.input_h, self.input_w = input_shape
        self.thre1 = self.params['thre1']
        self.thre2 = self.params['thre2']
        self.model = None
        self.gaussian_filtering = gaussian_filtering

    def load_model(self):
        self.model = self._create_model()
        print('Model loaded successfully')
        return self.model

    def _read_config(self):
        config = ConfigObj(self.config_path)
        param = config['param']
        model_id = param['modelID']
        model = config['models'][model_id]
        model['boxsize'] = int(model['boxsize'])
        model['stride'] = int(model['stride'])
        model['padValue'] = int(model['padValue'])
        param['octave'] = int(param['octave'])
        param['use_gpu'] = int(param['use_gpu'])
        param['starting_range'] = float(param['starting_range'])
        param['ending_range'] = float(param['ending_range'])
        param['scale_search'] = list(map(float, param['scale_search']))
        param['thre1'] = float(param['thre1'])
        param['thre2'] = float(param['thre2'])
        param['thre3'] = float(param['thre3'])
        param['mid_num'] = int(param['mid_num'])
        param['min_num'] = int(param['min_num'])
        param['crop_ratio'] = float(param['crop_ratio'])
        param['bbox_ratio'] = float(param['bbox_ratio'])
        param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])
        return param, model

    def _create_model(self):
        openpose_model = OpenPoseModelV2()
        openpose_raw = openpose_model.create_model()
        openpose_raw.load_weights(self.weights_path)

        input_tensor = tfkl.Input(shape=(self.input_h, self.input_w, 3))
        x = openpose_raw(input_tensor)
        hm = tf.image.resize(x[0], (self.input_h, self.input_w), 'bicubic')
        paf = tf.image.resize(x[1], (self.input_h, self.input_w), 'bicubic')

        if self.gaussian_filtering:
            gaussian_kernel = self._get_gaussian_kernel()
            depth_wise_gaussian_kernel = tf.expand_dims(
                tf.transpose(tf.keras.backend.repeat(gaussian_kernel, openpose_model.np_cm), perm=(0, 2, 1)), axis=-1)
            hm = tf.nn.depthwise_conv2d(hm,
                                        depth_wise_gaussian_kernel,
                                        [1, 1, 1, 1],
                                        'SAME')

        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        padded = tf.pad(hm, paddings)

        slice1 = tf.slice(padded, [0, 0, 1, 0], [-1, self.input_h, self.input_w, -1])
        slice2 = tf.slice(padded, [0, 2, 1, 0], [-1, self.input_h, self.input_w, -1])
        slice3 = tf.slice(padded, [0, 1, 0, 0], [-1, self.input_h, self.input_w, -1])
        slice4 = tf.slice(padded, [0, 1, 2, 0], [-1, self.input_h, self.input_w, -1])

        stacked = tf.stack([hm >= slice1, hm >= slice2, hm >= slice3, hm >= slice4, hm >= self.thre1], axis=-1)
        binary_hm = tf.reduce_all(stacked, axis=-1)

        masked_hm = tf.multiply(tf.cast(binary_hm, tf.float32), hm)

        model = tfk.Model(input_tensor, [paf, masked_hm])
        return model

    @staticmethod
    def _get_gaussian_kernel(mean=0, sigma=3):
        size = sigma * 3
        d = tfb.distributions.Normal(mean, sigma)
        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
        gauss_kernel = tf.einsum('i,j->ij', vals, vals)
        return gauss_kernel


class OpenPoseModelV2:

    def __init__(self,
                 input_shape=(None, None, 3),
                 paf_stages=4,
                 cm_stages=2,
                 np_paf=52,
                 np_cm=26):
        """OpenPose model definition proposed in arXiv:1812.08008v2

        :param input_shape: (height, width, n_channels)
        :param paf_stages: int - number of blocks for part affinity fields
        :param cm_stages: int - number of blocks for key-point heat-maps, i.e. confidence maps
        :param np_paf: int - number of channels for part affinity fields
        :param np_cm: int - number of channels for confidence maps, i.e. number of joints plus 1 for background

        Note: Input image must be RGB(0, 255)
        """

        self.input_shape = input_shape
        self.paf_stages = paf_stages
        self.cm_stages = cm_stages
        self.np_paf = np_paf
        self.np_cm = np_cm

    def create_model(self):
        input_tensor = tfkl.Input(self.input_shape)  # Input must be RGB and (0, 255
        normalized_input = tfkl.Lambda(lambda x: x / 256 - 0.5)(input_tensor)  # [-0.5, 0.5]

        # VGG
        initial_features = self._vgg_block(normalized_input)

        # PAF blocks
        paf_out = self._paf_block(initial_features, 0, 96, 256)

        for paf_stage in range(1, self.paf_stages):
            concat = tfkl.Concatenate(axis=-1, name='concat_stage{}_L2'.format(paf_stage))([initial_features, paf_out])
            paf_out = self._paf_block(concat, paf_stage, 128, 512)

        # Confidence maps blocks
        cm_input = tfkl.Concatenate(axis=-1, name='concat_stage0_L1')([initial_features, paf_out])
        cm_out = self._cm_block(cm_input, 0, 96, 256)

        for cm_stage in range(1, self.cm_stages):
            concat = tfkl.Concatenate(axis=-1, name='concat_stage{}_L1'.format(cm_stage))([initial_features, cm_out, paf_out])
            cm_out = self._cm_block(concat, cm_stage, 128, 512)

        model = tfk.Model(input_tensor, [cm_out, paf_out])
        return model

    def _paf_block(self, x, stage, n_kernels, n_pw_kernels):
        x = self._res_conv(x, stage, 1, n_kernels, 'L2')
        x = self._res_conv(x, stage, 2, n_kernels, 'L2')
        x = self._res_conv(x, stage, 3, n_kernels, 'L2')
        x = self._res_conv(x, stage, 4, n_kernels, 'L2')
        x = self._res_conv(x, stage, 5, n_kernels, 'L2')
        x = self._conv(x, n_pw_kernels, 1, 'Mconv6_stage{}_L2'.format(stage))
        x = self._prelu(x, 'Mprelu6_stage{}_L2'.format(stage))
        x = self._conv(x, self.np_paf, 1, 'Mconv7_stage{}_L2'.format(stage))
        return x

    def _cm_block(self, x, stage, n_kernels, n_pw_kernels):
        x = self._res_conv(x, stage, 1, n_kernels, 'L1')
        x = self._res_conv(x, stage, 2, n_kernels, 'L1')
        x = self._res_conv(x, stage, 3, n_kernels, 'L1')
        x = self._res_conv(x, stage, 4, n_kernels, 'L1')
        x = self._res_conv(x, stage, 5, n_kernels, 'L1')
        x = self._conv(x, n_pw_kernels, 1, 'Mconv6_stage{}_L1'.format(stage))
        x = self._prelu(x, 'Mprelu6_stage{}_L1'.format(stage))
        x = self._conv(x, self.np_cm, 1, 'Mconv7_stage{}_L1'.format(stage))
        return x

    def _res_conv(self, x, stage, block, n_kernels, block_type):
        conv_name = 'Mconv{}_stage{}_{}_'.format(block, stage, block_type)
        activation_name = 'Mprelu{}_stage{}_{}_'.format(block, stage, block_type)
        out1 = self._conv(x, n_kernels, 3, conv_name + str(0))
        out1 = self._prelu(out1, activation_name + str(0))

        out2 = self._conv(out1, n_kernels, 3, conv_name + str(1))
        out2 = self._prelu(out2, activation_name + str(1))

        out3 = self._conv(out2, n_kernels, 3, conv_name + str(2))
        out3 = self._prelu(out3, activation_name + str(2))

        out = tfkl.Concatenate(axis=-1, name=conv_name + 'concat')([out1, out2, out3])
        return out

    def _vgg_block(self, x):
        # Block 1
        x = self._conv(x, 64, 3, "conv1_1")
        x = self._relu(x, 'relu1_1')
        x = self._conv(x, 64, 3, "conv1_2")
        x = self._relu(x, 'relu1_2')
        x = self._pooling(x, 2, 2, "pool1_stage1")

        # Block 2
        x = self._conv(x, 128, 3, "conv2_1")
        x = self._relu(x, 'relu2_1')
        x = self._conv(x, 128, 3, "conv2_2")
        x = self._relu(x, 'relu2_2')
        x = self._pooling(x, 2, 2, "pool2_stage1")

        # Block 3
        x = self._conv(x, 256, 3, "conv3_1")
        x = self._relu(x, 'relu3_1')
        x = self._conv(x, 256, 3, "conv3_2")
        x = self._relu(x, 'relu3_2')
        x = self._conv(x, 256, 3, "conv3_3")
        x = self._relu(x, 'relu3_3')
        x = self._conv(x, 256, 3, "conv3_4")
        x = self._relu(x, 'relu3_4')
        x = self._pooling(x, 2, 2, "pool3_stage1")

        # Block 4
        x = self._conv(x, 512, 3, "conv4_1")
        x = self._relu(x, 'relu4_1')
        x = self._conv(x, 512, 3, "conv4_2")
        x = self._prelu(x, 'prelu4_2')

        # Additional non vgg layers
        x = self._conv(x, 256, 3, "conv4_3_CPM")
        x = self._prelu(x, 'prelu4_3_CPM')
        x = self._conv(x, 128, 3, "conv4_4_CPM")
        x = self._prelu(x, 'prelu4_4_CPM')
        return x

    @staticmethod
    def _conv(x, nf, ks, name):
        out = tfkl.Conv2D(nf, (ks, ks), padding='same', name=name)(x)
        return out

    @staticmethod
    def _relu(x, name):
        return tfkl.Activation('relu', name=name)(x)

    @staticmethod
    def _prelu(x, name):
        return tfkl.PReLU(shared_axes=[1, 2], name=name)(x)

    @staticmethod
    def _pooling(x, ks, st, name):
        x = tfkl.MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
        return x



