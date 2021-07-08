import requests

import numpy as np
import cv2


class Detection:

    def __init__(self,
                 key_points,
                 transformed_candidate,
                 person_subset,
                 confidences,
                 bbox):
        self.key_points = key_points
        self.transformed_candidate = transformed_candidate
        self.person_subset = person_subset
        self.confidences = confidences
        self.bbox = bbox
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 255, 0], [170, 255, 0], [85, 255, 0],
                       [0, 255, 0], [0, 255, 85], [0, 255, 170],
                       [0, 255, 255], [0, 170, 255], [0, 85, 255],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [255, 0, 255], [255, 0, 170], [255, 0, 85],
                       [255, 170, 85], [255, 170, 170], [255, 170, 255],
                       [255, 85, 85], [255, 85, 170], [255, 85, 255],
                       [170, 170, 170]]
        random_colors = np.random.randint(0, 255, (50, 3)).tolist()
        self.colors.extend(random_colors)

        self.id = None
        self.head_bbox = None
        self.person_representation = None
        self.face_representation = None

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


class BoundingBox:

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_left = x_min
        self.y_up = y_min
        self.width = x_max - x_min
        self.height = y_max - y_min
        self.data = [x_min, y_min, x_max, y_max]
        self.diag_len = np.sqrt(np.power(x_max - x_min, 2) + np.power(y_max - y_min, 2))


class Drawer:

    def __init__(self,
                 colors,
                 n_limbs,
                 connections,
                 stick):

        """Give colors and connections from hyper_config."""

        self.colors = colors
        self.n_limbs = n_limbs
        self.connections = connections
        self.stick = stick

    def draw_kps(self, img, kps):
        for i, kp in enumerate(kps):
            if isinstance(kp, np.ndarray):
                if not np.any(np.isnan(kp)):
                    color = self.colors[i]
                    self._draw_kp(img, kp, color)
                    # cv2.circle(img, (int(kp[0]), int(kp[1])), self.stick, self.colors[i], thickness=-1)
            else:
                if kp is not None:
                    color = self.colors[i]
                    self._draw_kp(img, kp, color)
                    # cv2.circle(img, (int(kp[0]), int(kp[1])), self.stick, self.colors[i], thickness=-1)

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

    def _draw_kp(self, img, kp, color):
        cv2.circle(img, (int(kp[0]), int(kp[1])), self.stick, color, thickness=-1)

    @staticmethod
    def _draw_arc(image, center, radius, start_angle, end_angle, color):
        overlay = image.copy()

        axes = (radius, radius)
        angle = 0
        thickness = -1

        cv2.ellipse(overlay, (int(center[0]), int(center[1])), axes, angle, start_angle, end_angle, color, thickness)
        drawed = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
        return drawed


class Downloader:

    def __init__(self):
        self.chunk_size = 32768
        self.url = "https://docs.google.com/uc?export=download"

    def download_file_from_google_drive(self, file_id, destination):
        session = requests.Session()

        response = session.get(self.url, params={'id': file_id}, stream=True)
        token = self.get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(self.url, params=params, stream=True)

        self.save_response_content(response, destination)

    def save_response_content(self, response, destination):

        with open(destination, "wb") as f:
            for chunk in response.iter_content(self.chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    @staticmethod
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None
