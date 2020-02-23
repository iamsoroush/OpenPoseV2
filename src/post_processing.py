import os

import numpy as np
import pandas as pd
import cv2
from scipy import fftpack

from .models import OpenPoseV2
from .config import HyperConfig, FeatureExtractorConfig


class PostProcessor:

    def __init__(self,
                 fps,
                 moving_average_len=3,
                 overlap=0.5,
                 window_len=2):

        """Post processing for pose sequences and corresponding errors.

        Note: window length for calculating epoch information is 2 * fps

        :arg fps: frame rate.
        :arg moving_average_len: for each pose error feature, a smoothing window of this length will be added.
        :arg overlap: overlap between epochs in seconds.
        :arg window_len: each epoch's length in seconds."""

        self.fe_config = FeatureExtractorConfig()
        self.fps = fps
        self.overlap = int(self.fps * overlap)
        self.window_len = int(self.fps * window_len)
        self.moving_average_len = moving_average_len

    def get_person_errors(self, person):

        """Extracts person errors' absolute values, interpolates nan elements and returns
            moving-averaged signals if moving_av==True.

        :param moving_av: whether to do moving-average on signals.
        :arg person: an instance of Person class.
        """

        person_frames = list(person.detections.keys())
        n_features = len(list(person.detections.values())[0].pose_error)
        errors = np.empty((max(person_frames), n_features))
        errors[:] = np.nan
        for f_id in person_frames:
            #             if not absolute:
            #                 errors[f_id - 1] = person.detections[f_id].pose_error
            #             else:
            errors[f_id - 1] = np.abs(np.array(person.detections[f_id].pose_error, dtype=np.float))
        # for i in range(max(person_frames)):
        #     if i not in person_frames:
        #         errors.append([np.nan] * n_features)
        #     else:
        #         errors.append(person.detections[i].pose_error)

        points_comb_str = self.fe_config.points_comb_str

        columns = ['-'.join(i) for i in points_comb_str]
        df = pd.DataFrame(errors, columns=columns, dtype=np.float32)
        int_df = df.interpolate(axis=0, limit=int(self.fps))
        if self.moving_average_len is not None:
            filt = np.ones((self.moving_average_len,)) / self.moving_average_len
            out = int_df.apply(np.convolve, args=(filt, 'full'), axis=0)
        else:
            out = int_df
        return out

    def get_epoch_information(self, errors_df, peak_freq=10):
        t_steps = int(errors_df.shape[0])
        epoched_errors = list()
        index = list()
        for i, start_ind in enumerate(range(0, t_steps, self.overlap)):
            end_ind = start_ind + self.window_len
            if end_ind >= t_steps:
                break
            err = errors_df.iloc[start_ind: end_ind]
            filtered = self._filter_high_freq(err, peak_freq, self.fps)
            epoched_errors.append(filtered.mean(axis=0))

            #             epoched_errors.append(errors_df.iloc[start_ind: end_ind].mean().values)
            # index.append('epoch_' + str(i + 1))
            start_sec = start_ind / self.fps
            end_sec = end_ind / self.fps
            index.append('{:2.2f}s --> {:2.2f}s'.format(start_sec, end_sec))
        epochs_df = pd.DataFrame(epoched_errors, columns=errors_df.columns, index=index)
        return epochs_df

    @staticmethod
    def _filter_high_freq(epoch, peak_freq, fps):
        sig_fft = fftpack.rfft(epoch.T)

        time_step = 1 / fps
        sample_freq = fftpack.rfftfreq(epoch.shape[0], d=time_step)

        high_freq_fft = sig_fft.copy()
        high_freq_fft[:, np.abs(sample_freq) > peak_freq] = 0
        filtered_sig = fftpack.irfft(high_freq_fft)
        return filtered_sig.T

    def _get_frame_inds_for_epoch(self, epoch_name):
        epoch_number = epoch_name.split('_')[-1]
        start = int((epoch_number - 1) * self.overlap)
        end = start + self.window_len
        return start, end


class TemplatePose:

    def __init__(self, openpose=None):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        template_dir = os.path.join(self.file_dir, 'template')
        if not os.path.isdir(template_dir):
            os.mkdir(template_dir)
        self.template_kp_path = os.path.join(template_dir, 'template.npy')
        self.template_img_path = os.path.join(template_dir, 'template.jpg')
        self.standing_img_path = os.path.join(template_dir, 'standing.jpg')

        self.template_kp, self.template_img, self.max_error_radius, self.standing_img = self._get_template(openpose)
        print('Template pose loaded.')

        self.error_color = (255, 0, 0)
        self.hc = HyperConfig()

    def get_error_template(self, errors):

        """:arg errors: {'joint_1': error_1, ... , 'joints_n': error_n}"""

        drawed = self.standing_img.copy()
        for kp_name, kp_err in errors.items():
            drawed = self._draw_error(kp_err, drawed, kp_name)
        return drawed

    def _draw_error(self, error, template, kp_name):
        if np.any(np.isnan(error)):
            return template
        if error < self.hc.error_th:
            return template
        radius = int(self.max_error_radius * error / 360)
        kp = self.template_kp[kp_name]
        overlay = template.copy()
        cv2.circle(overlay, (int(kp[0]), int(kp[1])), radius, self.error_color, thickness=-1)
        drawed = cv2.addWeighted(template, 0.4, overlay, 0.6, 0)
        return drawed

    def _get_template(self, openpose=None):
        standing_img = cv2.cvtColor(cv2.imread(self.standing_img_path), cv2.COLOR_BGR2RGB)
        if not os.path.exists(self.template_kp_path) or not os.path.exists(self.template_img_path):
            print('generating template ...')
            assert isinstance(openpose, OpenPoseV2), 'You must provide an instance of OpenPoseV2.'

            detection = openpose.get_detections(standing_img)[0]
            h, w = standing_img.shape[:2]
            template = np.zeros((h, w, 3))
            template_img = openpose.draw_detection(template, detection, False, False, True, True)
            self._save_template(detection, template_img)

        template_kp = np.load(self.template_kp_path, allow_pickle=True).item()
        max_radius = template_kp.pop('max_radius')
        template_img = cv2.cvtColor(cv2.imread(self.template_img_path), cv2.COLOR_BGR2RGB)
        return template_kp, template_img, max_radius, standing_img

    def _save_template(self, detection, template_img):
        template_kp = dict(zip(self.hc.kp_names[:-1], detection.key_points))
        max_radius = detection.bbox.diag_len // 10
        template_kp['max_radius'] = max_radius
        np.save(self.template_kp_path, template_kp)
        cv2.imwrite(self.template_img_path, template_img)
