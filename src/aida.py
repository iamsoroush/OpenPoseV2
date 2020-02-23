import os
import pickle

import cv2
from tqdm import tqdm
import numpy as np

from .tracker import SortTracker
from .utils import Track
from .config import OpenPoseV2Config, HyperConfig
from .models import OpenPoseV2


class AIDA:

    """Main wrapper for AI-based dance analysing."""

    def __init__(self, openpose_config, hyper_config, verbose):
        if openpose_config is None:
            openpose_config = OpenPoseV2Config()
        if hyper_config is None:
            hyper_config = HyperConfig()
        self.openpose = OpenPoseV2(openpose_config, hyper_config, verbose=verbose)

    def _get_sample_persons(self, video_path, to_run=5):
        vidcap = cv2.VideoCapture(video_path)

        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fc = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        if n_frames is None:
            n_frames = fc

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        outvid = cv2.VideoWriter(os.path.join(out_dir, 'out.mp4'), fourcc, fps, (width, height))

        tracker = SortTracker()
        tracks = dict()

        with tqdm(total=n_frames) as pbar:
            for i in range(n_frames):
                ret, frame = vidcap.read()
                assert ret, 'Debia!'

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = self.openpose.get_detections(frame)
                tracker.update(detections)
                if not np.any(detections):
                    pbar.update(1)
                    continue
                p_ids = list(tracks.keys())
                drawed = frame.copy()
                for detection in detections:
                    det_id = detection.id
                    if det_id in p_ids:
                        tracks[det_id].update(detection, i + 1)
                    else:
                        p = Track(det_id)
                        p.update(detection, i + 1)
                        tracks[det_id] = p
                    if det_id == leader_id:
                        target_features = None
                    else:
                        target_features = [det.pose_features for det in detections if det.id == leader_id][0]
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

        with open(os.path.join(out_dir, 'tracks.pkl'), 'wb') as output:
            pickle.dump(tracks, output, pickle.HIGHEST_PROTOCOL)

        print('Wrote results on ', out_dir)

    def run_on_webcam(self):
        pass