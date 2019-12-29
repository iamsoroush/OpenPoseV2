import numpy as np


class Person:

    """Base class for tracking trajectories.

    Access to trjectory using self.detections.

    :param detections: dict('frame_indx': Detection)"""

    def __init__(self, p_id):
        self.id = p_id
        self.detections = dict()

    def update(self, detection, frame_indx):
        assert isinstance(detection, Detection), 'detection must be an object of Detection class.'
        self.detections[frame_indx] = detection
        detection.id = self.id


class Detection:

    def __init__(self,
                 key_points,
                 transformed_candidate,
                 person_subset,
                 confidence,
                 bbox,
                 pose_features):
        self.key_points = key_points
        self.transformed_candidate = transformed_candidate
        self.person_subset = person_subset
        self.confidence = confidence
        self.bbox = bbox
        self.pose_features = pose_features
        self.pose_error = None
        self.id = None
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 255, 0], [170, 255, 0], [85, 255, 0],
                       [0, 255, 0], [0, 255, 85], [0, 255, 170],
                       [0, 255, 255], [0, 170, 255], [0, 85, 255],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [255, 0, 255], [255, 0, 170], [255, 0, 85],
                       [255, 170, 85], [255, 170, 170], [255, 170, 255],
                       [255, 85, 85], [255, 85, 170], [255, 85, 255],
                       [170, 170, 170]]

    def get_crop(self, img):
        """:arg img: RGB(0, 255) image."""

        person = img[self.bbox.y_up: self.bbox.y_up + self.bbox.height,
                     self.bbox.x_left: self.bbox.x_left + self.bbox.width,
                     :]
        return person

    def get_bbox_color(self):
        if self.id is None:
            return 0, 255, 0
        else:
            return self.colors[int(self.id)]

    def calc_error(self, ref_features):
        pass


class BoundingBox:

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_left = x_min
        self.y_up = y_min
        self.width = x_max - x_min
        self.height = y_max - y_min
        self.data = [x_min, y_min, x_max, y_max]


class FeatureExtractor:
    """Based on COCO keypoints, this feature extractor extracts features from extracted body keypoints."""

    def __init__(self, config):
        self.points_comb = config.points_comb

    def generate_features(self, keypoints):
        features = list()
        for comb in self.points_comb:
            feature = None

            a = keypoints[comb[0]]
            b = keypoints[comb[1]]
            c = keypoints[comb[2]]

            if (a is not None) and (b is not None) and (c is not None):
                feature = self._compute_angle(np.array(a),
                                              np.array(b),
                                              np.array(c))
            features.append(feature)
        return np.array(features)

    @staticmethod
    def _compute_angle(a, b, c):
        """Computes angle on point 'b'."""
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / ((np.linalg.norm(ba) * np.linalg.norm(bc)) + 0.0001)
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        return angle