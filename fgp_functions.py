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


# def feature_con(*args):
#     feature_vector = np.concatenate(args, axis=0)
#     return feature_vector
def feature_con(*args):
    vector_number = len(args)
    if vector_number == 2:
        vector1 = args[0]
        vector2 = args[1]
        for i in range(0, len(vector1)):
            vector1[i] = conVector(vector1[i])
        feature_vector1 = np.concatenate(vector1, axis=0)
        for j in range(0, len(vector2)):
            vector2[j] = conVector(vector2[j])
        feature_vector2 = np.concatenate(vector2, axis=0)
        feature_vector = np.concatenate((feature_vector1, feature_vector2), axis=0)
    elif vector_number == 3:
        vector1 = args[0]
        vector2 = args[1]
        vector3 = args[2]
        for i in range(0, len(vector1)):
            vector1[i] = conVector(vector1[i])
        feature_vector1 = np.concatenate(vector1, axis=0)
        for j in range(0, len(vector2)):
            vector2[j] = conVector(vector2[j])
        feature_vector2 = np.concatenate(vector2, axis=0)
        for k in range(0, len(vector3)):
            vector3[k] = conVector(vector3[k])
        feature_vector3 = np.concatenate(vector3, axis=0)
        feature_vector = np.concatenate((feature_vector1, feature_vector2, feature_vector3), axis=0)
    else:
        vector1 = args[0]
        vector2 = args[1]
        vector3 = args[2]
        vector4 = args[3]
        for i in range(0, len(vector1)):
            vector1[i] = conVector(vector1[i])
        feature_vector1 = np.concatenate(vector1, axis=0)
        for j in range(0, len(vector2)):
            vector2[j] = conVector(vector2[j])
        feature_vector2 = np.concatenate(vector2, axis=0)
        for k in range(0, len(vector3)):
            vector3[k] = conVector(vector3[k])
        feature_vector3 = np.concatenate(vector3, axis=0)
        for m in range(0, len(vector4)):
            vector4[m] = conVector(vector4[m])
        feature_vector4 = np.concatenate(vector4, axis=0)
        feature_vector = np.concatenate((feature_vector1, feature_vector2,
                                            feature_vector3, feature_vector4), axis=0)
    return feature_vector


def feature_con2(vector1, vector2):
    for i in range(0, len(vector1)):
        vector1[i] = conVector(vector1[i])
    feature_vector1 = np.concatenate(vector1, axis=0)
    feature_vector = np.concatenate((feature_vector1, vector2), axis=0)
    return feature_vector


# square
def regionS(left, x, y, windowSize):
    width, height = left.shape
    # region_result = []
    if (x + windowSize) < width and (y + windowSize) < height:
        region = left[x:(x + windowSize), y:(y + windowSize)]
    elif (x + windowSize) > width and (y + windowSize) < height:
        region = left[x:width, y:(y + windowSize)]
    elif (x + windowSize) < width and (y + windowSize) > height:
        region = left[x:(x + windowSize), y:height]
    else:
        region = left[x:width, y:height]
    # print('square', region.shape)
    # region_result.append(np.array(region))
    return region


# rectangle
def regionR(left, x, y, window_size1, window_size2):
    width, height = left.shape
    # region_result = []
    if (x + window_size1) < width and (y + window_size2) < height:
        region = left[x:(x + window_size1), y:(y + window_size2)]
    elif (x + window_size1) > width and (y + window_size2) < height:
        region = left[x:width, y:(y + window_size2)]
    elif (x + window_size1) < width and (y + window_size2) > height:
        region = left[x:(x + window_size1), y:height]
    else:
        region = left[x:width, y:height]
    # print('rectangle', region.shape)
    # region_result.append(np.array(region))
    return region


def featureMeanStd(region):
    #print(region)
    mean=np.mean(region)
    std=np.std(region)
    #print(mean,std)
    return mean,std


def featureEx_raw(image):
    features = []
    feature = np.zeros(20)
    width, height = image.shape
    width1 = int(width/2)
    height1 = int(height/2)
    width2 = int(width/4)
    height2 = int(height/4)
    #A1B1C1D1
    feature[0], feature[1] = featureMeanStd(image)
    #A1E1OG1
    feature[2], feature[3] = featureMeanStd(image[0:width1, 0:height1])
    #E1B1H1O
    feature[4], feature[5] = featureMeanStd(image[0:width1, height1:height])
    #G1OF1D1
    feature[6], feature[7] = featureMeanStd(image[width1:width, 0:height1])
    #OH1C1F1
    feature[8], feature[9] = featureMeanStd(image[width1:width, height1:height])
    #A2B2C2D2
    feature[10], feature[11] = featureMeanStd(image[width2:(width2+width1), height2:(height1+height2)])
    #G1H1
    feature[12], feature[13] = featureMeanStd(image[width1, :])
    #E1F1
    feature[14], feature[15] = featureMeanStd(image[:, height1])
    #G2H2
    feature[16], feature[17] = featureMeanStd(image[width1, height2:(height1+height2)])
    #E2F2
    feature[18], feature[19] = featureMeanStd(image[width2:(width2+width1), height1])
    features.append(np.array(feature))
    return feature


def featureEx_fun(image):
    feature=np.zeros((20))
    width,height=image.shape
    width1=int(width/2)
    height1=int(height/2)
    width2=int(width/4)
    height2=int(height/4)
    #A1B1C1D1
    feature[0],feature[1]=featureMeanStd(image)
    #A1E1OG1
    feature[2],feature[3]=featureMeanStd(image[0:width1,0:height1])
    #E1B1H1O
    feature[4],feature[5]=featureMeanStd(image[0:width1,height1:height])
    #G1OF1D1
    feature[6],feature[7]=featureMeanStd(image[width1:width,0:height1])
    #OH1C1F1
    feature[8],feature[9]=featureMeanStd(image[width1:width,height1:height])
    #A2B2C2D2
    feature[10],feature[11]=featureMeanStd(image[width2:(width2+width1),height2:(height1+height2)])
    #G1H1
    feature[12],feature[13]=featureMeanStd(image[width1,:])
    #E1F1
    feature[14],feature[15]=featureMeanStd(image[:,height1])
    #G2H2
    feature[16],feature[17]=featureMeanStd(image[width1,height2:(height1+height2)])
    #E2F2
    feature[18],feature[19]=featureMeanStd(image[width2:(width2+width1),height1])
    return feature


def featureEx(image):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        feature = featureEx_fun(x[0])
        features.append(np.array(feature))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                feature = featureEx_fun(x[j])
                features.append(np.array(feature))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


def hist_base(image):
    n_bins = 32
    hist, ax = np.histogram(image, n_bins, [0,1])
    return hist
def hist_raw(image):
    n_bins = 32
    hist, ax = np.histogram(image, n_bins, [0, 1])
    features = []
    x = hist
    features.append(np.array(x))
    return features


def hist(image):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    n_bins = 32
    if feature_number == 1:
        hist, ax = np.histogram(x[0], n_bins, [0, 1])
        features.append(np.array(hist))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                hist, ax = np.histogram(x[j], n_bins, [0, 1])
                features.append(np.array(hist))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features



def uniform_lbp_base(image):
    # uniform_LBP
    lbp = local_binary_pattern(image, P=8, R=1.5, method='nri_uniform')
    n_bins = 59
    hist, ax = np.histogram(lbp, n_bins, [0, 59])
    return hist
def uniform_lbp_raw(image):
    # uniform_LBP
    lbp = local_binary_pattern(image, P=8, R=1.5, method='nri_uniform')
    n_bins = 59
    hist, ax = np.histogram(lbp, n_bins, [0, 59])
    features = []
    x = hist
    features.append(np.array(x))
    return features


def uniform_lbp(image):
    x = image
    feature_number = len(x)
    # print('the number of input features, LBP', feature_number)
    flag = []
    features = []
    if feature_number == 1:
        lbp = local_binary_pattern(x[0], P=8, R=1.5, method='nri_uniform')
        n_bins = 59
        hist, ax = np.histogram(lbp, n_bins, [0, 59])
        features.append(np.array(hist))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                lbp = local_binary_pattern(x[j], P=8, R=1.5, method='nri_uniform')
                n_bins = 59
                hist, ax = np.histogram(lbp, n_bins, [0, 59])
                features.append(np.array(hist))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


def HoGFeatures(image):
    try:
        img,realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                    transform_sqrt=False, feature_vector=True)
        return realImage
    except:
        return image


def hog_features_patches(image,patch_size,moving_size):
    img = np.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
    hog_features = np.zeros((len(patch)))
    realImage=HoGFeatures(img)
    for i in range(len(patch)):
        hog_features[i] = np.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
    return hog_features


def global_hog_base(image):
    feature_vector = hog_features_patches(image, 4, 4)
    return feature_vector
def global_hog_raw(image):
    features = []
    x = hog_features_patches(image, 4, 4)
    features.append(np.array(x))
    # features = np.array(features)
    return features


def global_hog(image):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(hog_features_patches(x[0], 4, 4)))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(hog_features_patches(x[j], 4, 4)))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# def gau(left, si):
#     return gaussian(left,sigma=si)
def gau_raw(left, si):
    features = []
    x = gaussian(left, sigma=si)
    features.append(np.array(x))
    return features


def gau(image, si):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(gaussian(x[0],sigma=si))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(gaussian(x[j],sigma=si)))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# def gauD(left, si, or1, or2):
#     return ndimage.gaussian_filter(left,sigma=si, order=[or1,or2])
def gauD_raw(left, si, or1, or2):
    features = []
    x = ndimage.gaussian_filter(left,sigma=si, order=[or1,or2])
    features.append(np.array(x))
    return features


def gauD(image, si, or1, or2):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(ndimage.gaussian_filter(x[0],sigma=si, order=[or1,or2])))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(ndimage.gaussian_filter(x[j],sigma=si, order=[or1,or2])))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# def laplace(left):
#     return ndimage.laplace(left)
def laplace_raw(left):
    features = []
    x = ndimage.laplace(left)
    features.append(np.array(x))
    return features


def laplace(image):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(ndimage.laplace(x[0])))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(ndimage.laplace(x[j])))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# def gaussian_Laplace1(left):
#     return ndimage.gaussian_laplace(left,sigma=1)
def gaussian_Laplace1_raw(left):
    features = []
    x = ndimage.gaussian_laplace(left, sigma=1)
    features.append(np.array(x))
    return features


def gaussian_Laplace1(image):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(ndimage.gaussian_laplace(x[0], sigma=1)))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(ndimage.gaussian_laplace(x[j], sigma=1)))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# def gaussian_Laplace2(left):
#     return ndimage.gaussian_laplace(left,sigma=2)
def gaussian_Laplace2_raw(left):
    features = []
    x = ndimage.gaussian_laplace(left, sigma=2)
    features.append(np.array(x))
    return features


def gaussian_Laplace2(image):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(ndimage.gaussian_laplace(x[0], sigma=2)))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(ndimage.gaussian_laplace(x[j],sigma=2)))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# def sobelxy(left):
#     left = sobel(left)
#     feature_vector = conVector(left)
#     return feature_vector


# def sobelxy(image):
#     image = sobel(image)
#     return image
def sobelxy_raw(image):
    features = []
    x = sobel(image)
    features.append(np.array(x))
    return features


def sobelxy(image):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(sobel(x[0])))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(sobel(x[j])))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# def sobelx(image):
#     image = ndimage.sobel(image, axis=0)
#     return image
def sobelx_raw(image):
    features = []
    x = ndimage.sobel(image, axis=0)
    features.append(np.array(x))
    return features


def sobelx(image):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(ndimage.sobel(x[0], axis=0)))
    else:
        for i in range(0, feature_number):
            flag.append(np.array(random.randint(0, 1)))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(ndimage.sobel(x[j], axis=0)))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# def sobely(image):
#     image = ndimage.sobel(image, axis=1)
#     return image
def sobely_raw(image):
    features = []
    x = ndimage.sobel(image, axis=1)
    features.append(np.array(x))
    return features


def sobely(image):
    x = image
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(ndimage.sobel(x[0], axis=1)))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(ndimage.sobel(x[j], axis=1)))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# max filter
# def maxf(*args):
#     """
#     :type args: arguments and filter size
#     """
#     x = args[0]
#     if len(args) > 1:
#         size = args[1]
#     else:
#         size = 3
#     x = ndimage.maximum_filter(x, size)
#     return x
def max_raw(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size = 3
    features = []
    x = ndimage.maximum_filter(x, size)
    features.append(np.array(x))
    return features


def maxf(*args):
    if len(args) > 1:
        size = args[1]
    else:
        size = 3
    x = args[0]
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(ndimage.maximum_filter(x[0], size)))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(ndimage.maximum_filter(x[j], size)))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# median_filter
# def medianf(*args):
#     """
#     :type args: arguments and filter size
#     """
#     x = args[0]
#     if len(args) > 1:
#         size = args[1]
#     else:
#         size=3
#     x = ndimage.median_filter(x,size)
#     return x
def median_raw(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    features = []
    x = ndimage.median_filter(x, size)
    features.append(np.array(x))
    return features


def medianf(*args):
    if len(args) > 1:
        size = args[1]
    else:
        size = 3
    x = args[0]
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(ndimage.median_filter(x[0], size)))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(ndimage.median_filter(x[j], size)))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


# mean_filter
# def meanf(*args):
#     """
#     :type args: arguments and filter size
#     """
#     x = args[0]
#     if len(args) > 1:
#         size = args[1]
#     else:
#         size=3
#     x = ndimage.convolve(x, np.full((3, 3), 1 / (size * size)))
#     return x
def mean_raw(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    features = []
    x = ndimage.convolve(x, np.full((3, 3), 1 / (size * size)))
    features.append(np.array(x))
    return features


def meanf(*args):
    if len(args) > 1:
        size = args[1]
    else:
        size = 3
    x = args[0]
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(ndimage.convolve(x[0], np.full((3, 3), 1 / (size * size)))))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(ndimage.convolve(x[j], np.full((3, 3), 1 / (size * size)))))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


def min_raw(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    features = []
    x = ndimage.minimum_filter(x, size)
    features.append(np.array(x))
    return features


def minf(*args):
    if len(args) > 1:
        size = args[1]
    else:
        size = 3
    x = args[0]
    feature_number = len(x)
    flag = []
    features = []
    if feature_number == 1:
        features.append(np.array(ndimage.minimum_filter(x[0], size)))
    else:
        for i in range(0, feature_number):
            flag.append(random.randint(0, 1))
        for j in range(0, feature_number):
            if flag[j] == 1:
                features.append(np.array(ndimage.minimum_filter(x[j], size)))
    for m in range(0, feature_number):
        features.append(np.array(x[m]))
    return features


def classifier_selection(features, clf_id):
    #clf_id: 1=SVM,2=J48,3=RF
    if clf_id==1:
        clf = SVC(kernel='rbf')
    elif clf_id==2:
        clf = DecisionTreeClassifier(criterion='entropy') #J48->C4.5
    elif clf_id==3:
        clf=RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("无效分类器id")
    return (features,clf_id)

#测试
if __name__ == "__main__":
    a = [1, 2, 3]
    b = [4, 5, 6]
    print("Add:", vec_add(a, b))  # 应输出 [5,7,9]
    print("Conditional:", vec_conditional(a, [2,2,2], [10,10,10], [20,20,20]))  # 应输出 [10,20,20]
