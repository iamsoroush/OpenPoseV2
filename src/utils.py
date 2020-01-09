import numpy as np
import cv2


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
                 pose_features,
                 pose_feature_combs):
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
        self.pose_feature_combs = pose_feature_combs

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

    def calc_pose_error(self, target_features):
        errors = list()
        for i in range(len(self.pose_features)):
            f = self.pose_features[i]
            ft = target_features[i]
            if (f is not None) and (ft is not None):
                f_diff = ft - f
                if np.abs(f_diff) >= 180:
                    if f_diff > 0:
                        f_diff = -(360 - f_diff)
                    else:
                        f_diff = 360 - np.abs(f_diff)
            else:
                f_diff = None
            errors.append(f_diff)
        self.pose_error = np.array(errors, dtype=np.float)


class BoundingBox:

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_left = x_min
        self.y_up = y_min
        self.width = x_max - x_min
        self.height = y_max - y_min
        self.data = [x_min, y_min, x_max, y_max]
        self.diag_len = np.sqrt(np.power(x_max - x_min, 2) + np.power(y_max - y_min, 2))


class FeatureExtractor:
    """Based on COCO keypoints, this feature extractor extracts features from extracted body keypoints."""

    def __init__(self, points_comb):
        self.points_comb = points_comb

    def generate_features(self, keypoints):
        features = list()
        for comb in self.points_comb:
            a = keypoints[comb[0]]
            b = keypoints[comb[1]]
            c = keypoints[comb[2]]

            feature = None
            if (a is not None) and (b is not None) and (c is not None):
                feature = self._compute_angle(np.array(a),
                                              np.array(b),
                                              np.array(c))
            features.append(feature)
        return np.array(features)

    def _compute_angle(self, a, b, c):
        """Computes angle on point 'b'."""

        # a = np.array([p1[0], h - p1[1]])
        # b = np.array([p2[0], h - p2[1]])
        # c = np.array([p3[0], h - p3[1]])

        # ba = a - b
        # bc = c - b
        #
        # cosine_angle = np.dot(ba, bc) / ((np.linalg.norm(ba) * np.linalg.norm(bc)) + 0.0001)
        # angle = np.arccos(cosine_angle)
        # angle = np.degrees(angle)
        p1 = np.array([a[0], -a[1]])
        origin = np.array([b[0], -b[1]])
        p2 = np.array([c[0], -c[1]])

        angle = self._angle_between(p1 - origin, p2 - origin)

        return angle

    @staticmethod
    def _angle_between(p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))


class Drawer:

    def __init__(self,
                 feature_extractor,
                 colors,
                 n_limbs,
                 connections,
                 stick,
                 error_th,
                 error_color=(255, 0, 0)):

        """Give colors and connections from hyper_config."""

        assert isinstance(feature_extractor, FeatureExtractor)

        self.feature_extractor = feature_extractor
        self.colors = colors
        self.n_limbs = n_limbs
        self.connections = connections
        self.stick = stick
        self.error_th = error_th
        self.error_color = error_color

    def draw_pose_errors(self, img, detection):
        errors = detection.pose_error
        kps = detection.key_points
        org_h, org_w, _ = img.shape
        max_radius = detection.bbox.diag_len / 6

        for i, feature_error in enumerate(errors):
            if feature_error is not None:
                if np.abs(feature_error) >= self.error_th:
                    kp_2 = kps[self.feature_extractor.points_comb[i][1]]
                    kp_3 = kps[self.feature_extractor.points_comb[i][2]]
                    radius = int(max_radius * np.abs(feature_error) / 360)
                    # radius = int(max_radius)
                    if radius < 1:
                        continue
                    helper_point = [kp_2[0] + 1, kp_2[1]]

                    if feature_error > 0:
                        start_angle = self.feature_extractor._compute_angle(helper_point, kp_2, kp_3)
                        end_angle = start_angle + feature_error
                    else:
                        end_angle = self.feature_extractor._compute_angle(helper_point, kp_2, kp_3)
                        start_angle = end_angle + feature_error

                    # start_angle = self.fe._compute_angle(helper_point, kp_2, kp_3)
                    # end_angle = self.fe._compute_angle(helper_point, kp_2, kp_1)
                    img = self._draw_arc(img, kp_2, radius, end_angle, start_angle, self.error_color)
        return img

    @staticmethod
    def _draw_arc(image, center, radius, start_angle, end_angle, color):
        overlay = image.copy()

        axes = (radius, radius)
        angle = 0
        thickness = -1

        cv2.ellipse(overlay, (int(center[0]), int(center[1])), axes, angle, start_angle, end_angle, color, thickness)
        drawed = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
        return drawed

    def _draw_pose_feature_errors(self, drawed, detection):
        errors = detection.pose_error
        kps = detection.key_points
        org_h, org_w, _ = drawed.shape
        max_radius = org_h // 5

        for i, err in enumerate(errors):
            if err is not None:
                err = np.abs(err)
                if err >= self.error_th:
                    radius = int(max_radius * err / 360)
                    failure_overlay = drawed.copy()
                    kp = kps[self.feature_extractor.points_comb[i][1]]
                    cv2.circle(failure_overlay, (int(kp[0]), int(kp[1])), radius, self.error_color, thickness=-1)
                    drawed = cv2.addWeighted(drawed, 0.4, failure_overlay, 0.6, 0)
        return drawed

    def draw_kps(self, img, kps):
        for i, kp in enumerate(kps):
            if kp is not None:
                cv2.circle(img, (int(kp[0]), int(kp[1])), self.stick, self.colors[i], thickness=-1)

    def draw_connections(self, img, person, transformed_candidate):
        for i in range(self.n_limbs):
            index = person[np.array(self.connections[i])]
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
                                       (int(length / 2), self.stick),
                                       int(angle),
                                       0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
            img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
        return img
