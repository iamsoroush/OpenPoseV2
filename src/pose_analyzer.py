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

    def get_error_periods_for_joint(self, person, target_person, joint_name, threshold, fps):

        """Returns the error periods with error higher than the provided threshold in all epochs."""

        pass

    def get_errors(self, track, mode):

        """Returns mean/max errors for all joints and the drawed error template."""

        assert mode in ('max', 'mean'), "mode must be 'max' or 'mean'"

        person_errors = self.processor.get_person_errors(track)
        epoch_errors = self.processor.get_epoch_information(person_errors)
        central_joints = list(set([i.split('-')[1] for i in epoch_errors.columns]))
        errors = {i: [] for i in central_joints}

        if mode == 'mean':
            mean_errors = epoch_errors.mean(axis=0)
            for name, err in mean_errors.items():
                errors[name.split('-')[1]].append(err)
            for name, err in errors.items():
                errors[name] = np.mean(err)
        else:
            mean_errors = epoch_errors.max(axis=0)
            for name, err in mean_errors.items():
                errors[name.split('-')[1]].append(err)
            for name, err in errors.items():
                errors[name] = np.max(err)

        drawed = self.template_pose.get_error_template(errors)
        return errors, drawed

    def get_streaming_errors(self, track, window_per_second, save_dir):

        """Returns animation that streams errors of given track.

        :arg window_per_second: i.e. the fps of generated animation.
        :arg save_dir: generated animation will be saved as anim.mp4 in this directory.
        """

        person_errors = self.processor.get_person_errors(track)
        epoch_errors = self.processor.get_epoch_information(person_errors)
        central_joints = list(set([i.split('-')[1] for i in epoch_errors.columns]))

        templates = list()
        for period, error in epoch_errors.T.items():
            errors = {i: [] for i in central_joints}
            for name, err in error.items():
                errors[name.split('-')[1]].append(err)
            for name, err in errors.items():
                errors[name] = np.mean(err)
            temp_err = self.template_pose.get_error_template(errors)
            templates.append(temp_err)
        anim = self._animate(templates, window_per_second, save_dir)
        return anim

    def _animate(self, images, fps, save_dir=None):
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
        for img in images:
            im = plt.imshow(img, animated=True)
            ims.append([im])
        anim = animation.ArtistAnimation(fig,
                                         ims,
                                         interval=1000 / fps * 1.5,
                                         blit=True,
                                         repeat_delay=1000)

        if save_dir is not None:
            save_path = os.path.join(save_dir, 'anim.mp4')
            anim.save(save_path)
        return anim
