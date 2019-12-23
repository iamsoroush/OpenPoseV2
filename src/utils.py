import numpy as np


class BoundingBox:

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_left = x_min
        self.y_up = y_min
        self.width = x_max - x_min
        self.height = y_max - y_min


class Person:

    def __init__(self, keypoints, confidence, bbox, pose_features):
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = bbox
        self.pose_features = pose_features

    def get_person_data(self, img):
        """:arg img: RGB(0, 255) image."""

        person = img[self.bbox.y_up: self.bbox.y_up + self.bbox.height,
                     self.bbox.x_left: self.bbox.x_left + self.bbox.width,
                     :]
        return person


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