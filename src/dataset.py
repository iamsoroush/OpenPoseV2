import os

import numpy as np
import cv2
from pytube import YouTube


class PhysicalEducationDataset:

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.videos, self.video_paths = self._load_info()
        # self.videos = [item for item in os.listdir(self.dataset_dir) if item.endswith('.mp4')]
        # self.video_paths = [os.path.join(self.dataset_dir, item) for item in self.videos]

    def get_generator(self,
                      video_path,
                      batch_size,
                      resize_mode='none',
                      target_h=None,
                      target_w=None,
                      skip=0):

        """Returns a generator which in each iteration reads the next batch_size frames of the video and, resized them
        to desired size and returns RGB images as a numpy array with shape(batch_size, img_h, img_w, 3)

        Note: use a try-except to catch StopIteration:

            >> try:
            >>     next(gen)
            >> except StopIteration as e:
            >>     do_something()

        :arg video_path: complete path to video
        :arg batch_size: batch_size
        :arg target_h: output images' height
        :arg target_w: output images' width
        :arg resize_mode: none ==> no resizing, return original resolution.
                          preserve_aspect_ratio ==> resize with preserving aspect ratio, based on given target_h.
                          resize_with_pad ==> resize to (target_h, target_h) by preserving aspect ratio and padding
                           with value of pad_value.
                          absolute_resize ==> just resize to given target_h and target_w
        :arg skip: change the fps, e.g. if skip=1, reads one and skips one

        :returns generator, which yields an array of shape (batch_size, out_h, out_w, 3) for each iteration.
        """

        if target_h is None and target_w is None:
            resize_mode = 'none'

        vidcap = cv2.VideoCapture(video_path)
        width, height, fps, n_frames = self._get_vid_specs(vidcap)
        print('number of frames: {}'.format(n_frames))
        gen = self._generator(vidcap, n_frames, batch_size, target_h, target_w, resize_mode, skip)
        return gen

    def download_videos(self, urls_text_file_path):
        urls = self._get_urls(urls_text_file_path)
        # existing_vids = [item for item in os.listdir(self.dataset_dir) if item.endswith('.mp4')]

        for i, url in enumerate(urls):
            try:
                yt = YouTube(url)
            except Exception as e:
                print(e.args[0])
                continue
            YOUTUBE_ID = url.split('=')[-1]
            video_dir = os.path.join(self.dataset_dir, YOUTUBE_ID)
            stream = yt.streams.filter(file_extension='mp4', only_video=True, res='720p')[0]
            video_name = '{}.mp4'.format(YOUTUBE_ID)
            print('Downloading {} ...'.format(yt.title))
            try:
                stream.download(output_path=video_dir, filename=video_name.split('.')[0])
                # vid_path = stream.download(output_path=self.dataset_dir)
            except Exception as e:
                print(e.args[0])
                continue
            # self.videos.append(video_name)
            # self.video_paths.append(vid_path)
            # print('     saved as ', video_name)
        self.videos, self.video_paths = self._load_info()
        print('Done.')
        print('Videos: ')
        print(self.videos)

    def _load_info(self):
        videos = os.listdir(self.dataset_dir)
        video_paths = [os.path.join(self.dataset_dir, video_name, video_name + '.mp4') for video_name in videos]
        return videos, video_paths

    def _generator(self, vidcap, n_frames, batch_size, target_h, target_w, resize_mode, skip):
        n_frames_after_skip = int(n_frames / (skip + 1))
        steps = int(n_frames_after_skip // batch_size)
        for _ in range(steps):
            batch_data = list()
            for i in range(batch_size):
                for _ in range(skip + 1):
                    ret, frame = vidcap.read()
                resized = self._resize(frame, target_h, target_w, resize_mode)
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                batch_data.append(resized)
            yield np.array(batch_data)

    def _resize(self, frame, target_h, target_w, resize_mode):
        h, w = frame.shape[:2]

        if resize_mode == 'none':
            resized = frame
        elif resize_mode == 'preserve_aspect_ratio':
            scale = target_h / h
            resized = self._resize_aspect_ratio(frame, scale)
        elif resize_mode == 'resize_with_pad':
            resized = self._resize_with_pad(frame, target_h, 128)
        else:
            resized = self._absolute_resize(frame, target_h, target_w)

        return resized

    @staticmethod
    def _get_urls(file_path):
        video_urls = list()
        with open(file_path, 'r') as f:
            for line in f.readlines():
                if 'https' not in line:
                    continue
                video_urls.append(line[:-1])
        return video_urls

    @staticmethod
    def _get_vid_specs(vidcap):
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fc = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        return width, height, fps, fc

    @staticmethod
    def _resize_with_pad(img, target_res, pad_value):
        org_res = np.array(img.shape[:2])
        ratio = float(target_res) / max(org_res)
        new_size = (org_res * ratio).astype(int)

        resized = cv2.resize(img, (new_size[1], new_size[0]))

        delta_w = target_res - new_size[1]
        delta_h = target_res - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)

        left, right = delta_w // 2, delta_w - (delta_w // 2)

        resized_padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                            value=(pad_value, pad_value, pad_value))
        return resized_padded

    @staticmethod
    def _resize_aspect_ratio(img, scale_factor):
        org_h, org_w = np.array(img.shape[:2])
        new_h = (org_h * scale_factor).astype(int)
        new_w = (org_w * scale_factor).astype(int)

        resized = cv2.resize(img, (new_w, new_h))
        return resized

    @staticmethod
    def _absolute_resize(img, target_h, target_w):
        resized = cv2.resize(img, (target_w, target_h))
        return resized
