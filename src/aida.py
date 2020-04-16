import os
import pickle

import cv2
from tqdm import tqdm
import numpy as np

from .tracker import SortTracker
from .utils import Track
from .config import OpenPoseV2Config, HyperConfig, PoseCorrectionConfig
from .models import OpenPoseV2
from .pose_correction import PosePredictor, PoseCorrectionPreProcessor, PoseCorrector


class AIDA:

    """Main wrapper for AI-based dance analysing."""

    def __init__(self, openpose_config, hyper_config, pose_correction_config, correction_field=10, verbose=False):

        self.correction_field = correction_field

        if openpose_config is None:
            openpose_config = OpenPoseV2Config()
        if hyper_config is None:
            hyper_config = HyperConfig()
        if pose_correction_config is None:
            self.pose_correction_config = PoseCorrectionConfig()
        else:
            self.pose_correction_config = pose_correction_config

        self.openpose = OpenPoseV2(openpose_config, hyper_config, verbose=verbose)

        if self.correction_field is not None:
            self.pose_predictor = PosePredictor(self.pose_correction_config.weights_path,
                                                self.pose_correction_config.config_path)
            self.pp_model = self.pose_predictor.get_model()

            kp_mapper = {v: k for k, v in hyper_config.kp_mapper.items()}
            kp_mapper.pop('Background')
            self.pp_preprocessor = PoseCorrectionPreProcessor(kp_mapper, self.pose_predictor.config['invis_value'])

            self.pose_corrector = PoseCorrector(self.pose_correction_config.max_error_radius,
                                                self.correction_field)

        self.tracker = None

    def _get_sample_persons(self, video_path, to_run=5):
        vidcap = cv2.VideoCapture(video_path)

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        n_frames = int(to_run * fps)

        tracker = SortTracker()
        tracks = dict()

        with tqdm(total=n_frames) as pbar:
            for i in range(n_frames):
                ret, frame = vidcap.read()
                assert ret, 'Debia!'

                if np.any(frame):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = self.openpose.get_detections(frame)
                    tracker.update(detections)
                    if np.any(detections):
                        p_ids = list(tracks.keys())
                        for detection in detections:
                            det_id = detection.id
                            if det_id in p_ids:
                                tracks[det_id].update(detection, i + 1)
                            else:
                                p = Track(det_id)
                                p.update(detection, i + 1)
                                tracks[det_id] = p
                pbar.update(1)
        cv2.destroyAllWindows()
        vidcap.release()
        return tracks

    def run_on_video(self,
                     video_path,
                     out_dir,
                     leader_id,
                     draw_kps,
                     draw_limbs,
                     draw_bbox,
                     n_frames=None):

        vidcap = cv2.VideoCapture(video_path)

        width, height, fps, fc = self._get_vid_specs(vidcap)
        if n_frames is None:
            n_frames = int(fc)

        outvid = self._get_vid_writer(out_dir, fps, width, height)

        tracker = SortTracker()
        self.tracker = tracker
        tracks = dict()

        with tqdm(total=n_frames) as pbar:
            for frame_ind in range(n_frames):
                ret, frame = vidcap.read()
                assert ret, 'Debia!'

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = self._get_detections(frame)
                if detections is None:
                    pbar.update(1)
                    continue

                p_ids = list(tracks.keys())
                drawed = frame.copy()
                for detection in detections:
                    self._associate_det_to_track(detection, p_ids, tracks, frame_ind)
                    target_features = self._process_detection(detection, leader_id, detections)
                    drawed = self.openpose.draw_detection(drawed,
                                                          detection,
                                                          draw_kps=draw_kps,
                                                          draw_limbs=draw_limbs,
                                                          draw_bbox=draw_bbox,
                                                          target_features=target_features)

                outvid.write(cv2.cvtColor(drawed, cv2.COLOR_RGB2BGR))
                pbar.update(1)
        cv2.destroyAllWindows()
        vidcap.release()
        outvid.release()

        tracks_path = os.path.join(out_dir, 'tracks.pkl')
        with open(tracks_path, 'wb') as output:
            pickle.dump(tracks, output, pickle.HIGHEST_PROTOCOL)

        print('Wrote video on ', os.path.join(out_dir, 'out.mp4'))
        print('Wrote tracks on ', tracks_path)

    @staticmethod
    def _get_vid_specs(vidcap):
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fc = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        return width, height, fps, fc

    def _get_vid_writer(self, out_dir, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        outvid_path = os.path.join(out_dir, 'out.mp4')
        outvid = cv2.VideoWriter(outvid_path, fourcc, fps, (width, height))
        return outvid

    def _associate_det_to_track(self, detection, p_ids, tracks, frame_ind):
        det_id = detection.id
        if det_id in p_ids:
            track = tracks[det_id]
            if self.correction_field:
                self._correct_pose(track, detection)
            track.update(detection, frame_ind)
        else:
            track = Track(det_id)
            track.update(detection, frame_ind)
            tracks[det_id] = track
        return track

    @staticmethod
    def _process_detection(detection, leader_id, detections):
        det_id = detection.id
        target_features = None
        if not det_id == leader_id:
            my_list = [det.pose_features for det in detections if det.id == leader_id]
            if my_list:
                target_features = my_list[0]
                detection.update_pose_error(target_features)
        return target_features

    def _get_detections(self, frame):
        if not np.any(frame):
            self.tracker.update(list())
            return None
        detections = self.openpose.get_detections(frame)
        self.tracker.update(detections)
        if not np.any(detections):
            return None
        return detections

    def _correct_pose(self, track, detection, frame_ind):
        frame_inds = list(track.detections.keys())
        if len(frame_inds) >= self.correction_field:
            # recent_inds = frame_inds[-self.correction_field:]
            ind = frame_inds.index(frame_ind)
            recent_inds = frame_inds[ind - self.correction_field: ind]
            indxs = list(range(frame_ind - self.correction_field, frame_ind))
            if all(elem in recent_inds for elem in indxs):
                track_kps = np.array([track.detections[i].key_points for i in indxs])
                bboxes = np.array([track.detections[i].bbox.data for i in indxs])
                corrected_pose, _, _ = self.pose_corrector.correct_pose(track_kps,
                                                                        bboxes,
                                                                        self.pp_preprocessor,
                                                                        self.pp_model,
                                                                        detection)
                detection.key_points = corrected_pose

    def run_on_webcam(self):
        pass
