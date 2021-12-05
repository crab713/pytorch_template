import torch
import cv2
import os
import numpy as np
import imageio
from PIL import Image


def clearNoise(data):
    height = data.shape[0]
    width = data.shape[1]
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                data[i][j] = 255
                continue
            if data[i][j] == 0:
                num = 0
                for da in data[i - 1:i + 2, j - 1:j + 2]:
                    if da[0] > 0:
                        num += 1
                    if da[1] > 0:
                        num += 1
                    if da[2] > 0:
                        num += 1
                if num > 4:
                    data[i][j] = 255


def processImg(input_file, output_file=None) -> np.ndarray:
    """
    使图片格式统一，去除噪点
    :param input_file: input file path
    :param output_file: where to save processed image
    :return: np.array of image shape:[224, 224]
    """
    image = Image.open(input_file)
    img = image.convert("L")
    data = img.getdata()
    da = np.array(data, np.uint8)
    da[da <= 170] = 0
    da[da > 170] = 255
    da = da.reshape((70, 200))  # 图片原始尺寸为(70, 200)
    clearNoise(da)
    clearNoise(da)
    clearNoise(da)
    # da = np.resize(da, (32, 32))
    da = cv2.resize(da, (224, 224), interpolation=cv2.INTER_AREA)
    if output_file is not None:
        img1 = Image.fromarray(da)
        img1 = img1.convert('RGB')
        img1.save(output_file)

    return da


if __name__ == '__main__':
    origin_path = 'C:/Users/crab7/Desktop/ocr/data_set'
    train_path = origin_path + '/' + 'training'
    other_path = origin_path + '/' + 'other'

    img1_path = other_path + '/' + 'sfwmy_safg.jpg'
    img2_path = train_path + '/' + 'aaakw_FWMt7us2Z5.png'

    img1 = processImg(img1_path, 'test.png')

