import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

from .post_processing import PostProcessor, TemplatePose


class DanceAnalyzer:

    """This class alalyzes dancer's performance."""

    def __init__(self, fps, moving_average_len, overlap, window_len):
        self.processor = PostProcessor(fps=fps,
                                       moving_average_len=moving_average_len,
                                       overlap=overlap,
                                       window_len=window_len)
        self.template_pose = TemplatePose(openpose=None)

    def get_error_periods_for_joint(self, track, vidcap, joint_name, slow_motion=False, window_len=None, overlap=None):

        """Returns the error periods with error higher than the provided threshold in all epochs."""

        if window_len is None or overlap is None:
            processor = self.processor
        else:
            processor = PostProcessor(fps=self.processor.fps,
                                      moving_average_len=self.processor.moving_average_len,
                                      overlap=overlap,
                                      window_len=window_len)
        person_errors = processor.get_person_errors(track)
        epoch_errors = processor.get_epoch_information(person_errors)

        cols = epoch_errors.columns

        df = epoch_errors[[i for i in cols if i.split('-')[1] == joint_name]].max(axis=1)
        df = df.dropna().sort_values(ascending=False)
        threshold = df.mean()
        inds = df[df > threshold].index.tolist()
        starts = [float(i.split(' ')[0][:-1]) for i in inds]

        kp_index = self.template_pose.hc.kp_names.index(joint_name)
        gen = self._error_generator(starts, vidcap, track, processor.window_len, slow_motion, kp_index)
        return gen

    @staticmethod
    def _error_generator(starts, vidcap, track, window_len, slow_motion, kp_index):
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        for s in starts:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
            f_ind = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
            frames = list()
            for i in range(int(window_len)):
                ret, frame = vidcap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ind = f_ind + i
                if ind in list(track.detections.keys()):
                    detection = track.detections[ind]
                    (x_min, y_min, x_max, y_max) = detection.bbox.data
                    kp = detection.key_points[kp_index]
                    if not isinstance(kp, np.ndarray):
                        kp = np.array(kp, dtype=np.float)
                    if not np.any(np.isnan(kp)):
                        cv2.circle(frame,
                                   (int(kp[0]), int(kp[1])),
                                   int(detection.bbox.diag_len / 10),
                                   (255, 0, 0),
                                   thickness=5)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), detection.get_bbox_color(), 2, 5)
                frames.append(frame)
            if slow_motion:
                anim = _animate(frames, fps / 2, None)
            else:
                anim = _animate(frames, fps, None)
            yield anim

    # @staticmethod
    # def _error_generator(starts, vidcap, track, window_len):
    #     fps = vidcap.get(cv2.CAP_PROP_FPS)
    #     for s in starts:
    #         vidcap.set(cv2.CAP_PROP_POS_FRAMES, s * 1000)
    #         f_ind = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    #         frames = list()
    #         for i in range(int(window_len)):
    #             ret, frame = vidcap.read()
    #             ind = f_ind + i
    #             if ind in list(track.detections.keys()):
    #                 frames.append(track.detections[ind].get_crop(frame))
    #         yield _animate(frames, fps, None)

    def get_errors(self, track, mode):

        """Returns mean/max errors for all joints and the drawed error template.

        :arg track: Track object assigned to a person.
        :arg mode: whether to consider 'mean' or 'max' errors across epochs. Must be one of ('mean', 'max').

        :returns errors={'joint_1': error_1, ... , 'joints_n': error_n} and template with drawed errors on it.
        """

        assert mode in ('max', 'mean'), "mode must be 'max' or 'mean'"

        person_errors = self.processor.get_person_errors(track)
        epoch_errors = self.processor.get_epoch_information(person_errors)
        central_joints = list(set([i.split('-')[1] for i in epoch_errors.columns]))
        errors = {i: [] for i in central_joints}

        e = getattr(np, mode)(epoch_errors, axis=0)
        for name, err in e.items():
            errors[name.split('-')[1]].append(err)
        for name, err in errors.items():
            errors[name] = np.max(err)

        drawed = self.template_pose.get_error_template(errors)
        return errors, drawed

    def get_streaming_errors(self, track, epochs_per_second, save_dir=None):

        """Returns animation that streams errors of given track.

        :arg epochs_per_second: i.e. the fps of generated animation. Use epochs_per_second=1 and self.window_len=1 to
            have every second's information in every second of generated animation.
        :arg save_dir: generated animation will be saved as anim.mp4 in this directory.
        """

        person_errors = self.processor.get_person_errors(track)
        epoch_errors = self.processor.get_epoch_information(person_errors)
        central_joints = list(set([i.split('-')[1] for i in epoch_errors.columns]))

        templates = list()
        periods = list()
        for period, error in epoch_errors.T.items():
            errors = {i: [] for i in central_joints}
            for name, err in error.items():
                errors[name.split('-')[1]].append(err)
            for name, err in errors.items():
                errors[name] = np.max(err)
            temp_err = self.template_pose.get_error_template(errors)
            templates.append(temp_err)
            periods.append(period)
        anim = _animate(templates, epochs_per_second, save_dir, periods)
        return anim


def _animate(images, fps, save_dir, periods=None):
    fig = plt.figure(figsize=(14, 20))
    plt.axis('off')

    # im = plt.imshow(images[0])
    #
    # def animate_func(i):
    #     im.set_array(images[i])
    #     return [im]
    #
    # anim = animation.FuncAnimation(fig,
    #                                animate_func,
    #                                frames=len(images),
    #                                interval=1000 / fps * 1.5)
    ims = []
    for i, img in enumerate(images):
        if periods is not None:
            plt.title(periods[i])
        im = plt.imshow(img, animated=True)
        ims.append([im])
    anim = animation.ArtistAnimation(fig,
                                     ims,
                                     interval=1000 / fps,
                                     blit=True,
                                     repeat_delay=1000)

    if save_dir is not None:
        save_path = os.path.join(save_dir, 'anim.mp4')
        anim.save(save_path)
    return anim
