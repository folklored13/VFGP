import numpy as np
from pylab import *
from scipy import ndimage
import numpy as np

from skimage.filters import sobel
from skimage.filters import gabor
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import random
import pywt

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#LBPgray
def uniform_lbp_gray(image):
    #输入rgb图像要转成灰度
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    gray_uint8 = (gray * 255).astype(np.uint8)  # 新增转换
    lbp = local_binary_pattern(gray_uint8, P=8, R=1.5, method='nri_uniform')
    hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    return hist.astype(float).tolist()


#LBPrgb
def uniform_lbp_rgb(image):
    features = []
    for channel in range(3):  #RGB通道
        lbp = local_binary_pattern(image[:, :, channel], P=8, R=1.5, method='nri_uniform')
        hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
        features.extend(hist.tolist())
    return features


#小波特征
def wavelet_transform(image_with_lum):
    channels = [
        image_with_lum[:, :, 0],  # R
        image_with_lum[:, :, 1],  # G
        image_with_lum[:, :, 2],  # B
        0.299 * image_with_lum[:, :, 0] + 0.587 * image_with_lum[:, :, 1] + 0.114 * image_with_lum[:, :, 2]  # Luminance
    ]
    features = []
    for channel_data in channels:
        coeffs = pywt.wavedec2(channel_data, 'haar', level=3)
        channel_features = []

        # 近似系数 cA3
        stats = calculate_stats(coeffs[0])
        channel_features.extend(stats)  # 8

        # 每级细节系数 cH, cV, cD
        for level in range(1, 4):
            cH, cV, cD = coeffs[level]
            channel_features.extend(calculate_stats(cH))  # 8
            channel_features.extend(calculate_stats(cV))  # 8
            channel_features.extend(calculate_stats(cD))  # 8

        # 原始通道统计量
        stats_original = calculate_stats(channel_data)
        channel_features.extend(stats_original)  # 8

        # 当前：8 (cA) + 72 (3×3×8) + 8 (original) = 88

        center_region = channel_data[channel_data.shape[0] // 4:3 * channel_data.shape[0] // 4,
                        channel_data.shape[1] // 4:3 * channel_data.shape[1] // 4]
        stats_center = calculate_stats(center_region)
        channel_features.extend(stats_center)  # 8

        upper_region = channel_data[:channel_data.shape[0] // 2, :]
        stats_upper = calculate_stats(upper_region)
        channel_features.extend(stats_upper)  # 8

        features.extend(channel_features)

    return features

def calculate_stats(data):
    """计算8个统计量"""
    return [
        np.mean(data),
        np.std(data),
        entropy(data),
        kurtosis(data.flatten()),
        np.sum(data ** 2),
        np.mean(data ** 2),
        np.linalg.norm(data),
        skewness(data.flatten())
    ]

def vec_add(a, b):

    return list(np.add(np.asarray(a), np.asarray(b)))

def vec_sub(a, b):

    return list(np.subtract(np.asarray(a), np.asarray(b)))

def vec_mul(a, b):

    return list(np.multiply(np.asarray(a), np.asarray(b)))

def vec_div(a, b):

    b = np.asarray(b)
    b[b == 0] = 1e-7  # 防止除零
    return list(np.divide(np.asarray(a), b))

def vec_sin(a):

    return list(np.sin(np.asarray(a)))

def vec_cos(a):

    return list(np.cos(np.asarray(a)))

def vec_conditional(a, b, c, d):

    a = np.asarray(a)
    b = np.asarray(b)
    return list(np.where(a < b, c, d))


#辅助函数
def entropy(data):
    hist, _ = np.histogram(data, bins=256)
    prob = hist / hist.sum()
    return -np.sum(prob * np.log2(prob + 1e-7))


def kurtosis(data):
    return (np.mean((data - np.mean(data)) ** 4) / (np.std(data) ** 4)) - 3


def skewness(data):
    return np.mean((data - np.mean(data)) ** 3) / (np.std(data) ** 3)

#if操作
def conditional_op(a, b, c, d):
    a = np.asarray(a)
    b = np.asarray(b)
    return list(np.where(a < b, c, d))

def conVector(img):
    try:
        img_vector = np.concatenate(img, axis=0)
    except ValueError:
        img_vector = img
    return img_vector

