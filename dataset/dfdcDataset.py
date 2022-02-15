from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.utils import shuffle
import os
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
from base.dfdcTransform import DFDCTransform
import random

class DFDCDataset(Dataset):
    """json file type
    [{name, total_frame, labels},
     {name, total_frame, labels}
     ...]
    """
    def __init__(self, data_root : str, json_file : str):
        self.data_root = data_root
        self.json_file = json_file
        self.transform = DFDCTransform()

        self.real_videos = [] # {name, total_frame, label}
        self.fake_videos = []
        self.real_cursor = 0
        self.fake_cursor = 0

        self.frame_size = 16
        with open(json_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                info = eval(line)
                self.check_info(info)
                if info['label'] == 'REAL':
                    self.real_videos.append(info)
                elif info['label'] == 'FAKE':
                    self.fake_videos.append(info)
                else:
                    raise RuntimeError(
                        'line has inconsistent label: got {}, expected: REAL or FAKE'
                            .format(info['label']))


    def __len__(self) -> int:
        return len(self.real_videos) + len(self.fake_videos)

    def __getitem__(self, i):
        video_info = self.upsample_filter(i)
        frames = self.screen_video(video_info)
        
        inputs = np.array(frames)
        output = self.transform(inputs)

        label = 0 if video_info['label'] == 'REAL' else 1
        label = torch.tensor(int(label), dtype=torch.long)

        return {"image":output,
                "label":label}

    def upsample_filter(self, i : int) -> dict:
        """上采样规则
        Inputs:
            i: 该epoch第几次采样
        
        Returns:
            video_info: 所采样的视频信息
        """
        video_info = None
        if i%2 == 0:
            video_info = self.real_videos[self.real_cursor]
            self.real_cursor = (self.real_cursor + 1) % len(self.real_videos)
        else:
            video_info = self.fake_videos[self.fake_cursor]
            self.fake_cursor = (self.fake_cursor + 1) % len(self.fake_videos)

        return video_info

    def screen_video(self, video_info : dict) -> list:
        """对单个视频的取帧规则,
        Inputs:
            video_info: 待裁剪的视频信息: {name, total_frame, label}
        
        Outputs:
            output: 处理后的视频数组,shape=[frame[height, width, channel]]
        """
        keys = video_info.keys()
        if 'name' not in keys or 'total_frame' not in keys:
            raise RuntimeError(
                'need key not found, expect: name and total_frame, got: {}'.format(
                    video_info))

        capture = cv2.VideoCapture(self.data_root + video_info['name'])
        start_index = random.randint(0, video_info['total_frame'] - self.frame_size*18)

        frames = []
        for i in range(self.frame_size):
            index = start_index + i * 15
            capture.set(propId=cv2.CAP_PROP_POS_FRAMES, value=index) # 跳到指定帧
            hasframe, image1 = capture.read()
            frames.append(image1)
        return frames

    def check_info(self, info : dict) -> None:
        if info.get('label') is None:
            raise RuntimeError()
        
        if info.get('total_frame') is None:
            raise RuntimeError()

        if info.get('name') is None:
            raise RuntimeError()

if __name__ == '__main__':
    dataset = DFDCDataset('data/', 'train.txt')
    i = dataset[0]