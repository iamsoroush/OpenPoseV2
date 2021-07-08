import os
import pickle

import cv2
from tqdm import tqdm
import numpy as np

from .tracker import SortTracker, Track
from .config import OpenPoseV2Config, HyperConfig
from .models import OpenPoseV2


class DetecTracker:

    """Pose detector + Person tracker, which runs on a given video."""

    def __init__(self, openpose_config, hyper_config, verbose=False):

        if openpose_config is None:
            openpose_config = OpenPoseV2Config()
        if hyper_config is None:
            hyper_config = HyperConfig()

        self.openpose = OpenPoseV2(openpose_config, hyper_config, verbose=verbose)
        self.tracker = None

    def run_on_video(self,
                     video_path,
                     out_dir,
                     draw_kps=True,
                     draw_limbs=True,
                     draw_bbox=True,
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
                    drawed = self.openpose.draw_detection(drawed,
                                                          detection,
                                                          draw_kps=draw_kps,
                                                          draw_limbs=draw_limbs,
                                                          draw_bbox=draw_bbox)

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

    def _get_detections(self, frame):
        if not np.any(frame):
            self.tracker.update(list())
            return None
        detections = self.openpose.get_detections(frame)
        self.tracker.update(detections)
        if not np.any(detections):
            return None
        return detections

    @staticmethod
    def _get_vid_specs(vidcap):
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fc = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        return width, height, fps, fc

    @staticmethod
    def _get_vid_writer(out_dir, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        outvid_path = os.path.join(out_dir, 'out.mp4')
        outvid = cv2.VideoWriter(outvid_path, fourcc, fps, (width, height))
        return outvid

    @staticmethod
    def _associate_det_to_track(detection, p_ids, tracks, frame_ind):
        det_id = detection.id
        if det_id in p_ids:
            track = tracks[det_id]
            track.update(detection, frame_ind)
        else:
            track = Track(det_id)
            track.update(detection, frame_ind)
            tracks[det_id] = track
        return track

    def run_on_webcam(self):
        pass
