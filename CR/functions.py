import numpy as np
from pylab import *
from scipy import ndimage
import numpy as np

from skimage.feature import local_binary_pattern
import pywt
import math # Ensure math is imported for scalar functions

# LBPgray (Assuming it returns a list/numpy array of floats)
def uniform_lbp_gray(image):
    # 输入rgb图像要转成灰度
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, P=8, R=1.5, method='nri_uniform')
    # 调整bin范围以匹配nri_uniform的输出
    n_bins = int(lbp.max() + 1) if lbp.size > 0 else 59 # handles potential empty lbp if image is too small or uniform
    if n_bins > 59: n_bins = 59 # Ensure bins don't exceed max for uniform LBP (P=8 -> 59)
    if n_bins == 0: n_bins = 1 # Handle case where lbp.max() is -inf (e.g., empty image)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins)) # range should match bins if using n_bins from data
    # Use fixed bins as per common practice for comparison
    hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    return hist.astype(float).tolist() # Ensure float list


# LBPrgb (Assuming it returns a list/numpy array of floats)
def uniform_lbp_rgb(image):
    features = []
    for channel in range(3):  # RGB通道
        lbp = local_binary_pattern(image[:, :, channel], P=8, R=1.5, method='nri_uniform')
        n_bins = int(lbp.max() + 1) if lbp.size > 0 else 59
        if n_bins > 59: n_bins = 59
        if n_bins == 0: n_bins = 1
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        # Use fixed bins
        hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
        features.extend(hist.tolist())
    return features # Ensure float list


# 小波特征 (Assuming it returns a list/numpy array of floats)
# Keeping the original implementation that resulted in 416 features based on the total count in the paper
def wavelet_transform(image_with_lum):
    # Note: This implementation produced 416 features in the original code,
    # including stats on original, center, and upper regions,
    # which deviates from the paper's text description of 384 (coeffs) + 32 (original).
    # We keep this version as it aligns with the total feature count (652 = 59+177+416).
    channels = [
        image_with_lum[:, :, 0],  # R
        image_with_lum[:, :, 1],  # G
        image_with_lum[:, :, 2],  # B
        0.299 * image_with_lum[:, :, 0] + 0.587 * image_with_lum[:, :, 1] + 0.114 * image_with_lum[:, :, 2] # Luminance (assuming Luminance is channel 3 in image_with_lum)
    ]
    features = []
    for channel_data in channels:
        coeffs = pywt.wavedec2(channel_data, 'haar', level=3)
        channel_features = []

        # Approximation coefficients cA3
        stats = calculate_stats(coeffs[0])
        channel_features.extend(stats)

        # Details coefficients cH, cV, cD for levels 3, 2, 1
        for level in range(1, 4):
            cH, cV, cD = coeffs[level]
            channel_features.extend(calculate_stats(cH))
            channel_features.extend(calculate_stats(cV))
            channel_features.extend(calculate_stats(cD))

        # Original channel data stats
        stats_original = calculate_stats(channel_data)
        channel_features.extend(stats_original)

        # Stats on center region
        center_region = channel_data[channel_data.shape[0] // 4:3 * channel_data.shape[0] // 4,
                                     channel_data.shape[1] // 4:3 * channel_data.shape[1] // 4]
        stats_center = calculate_stats(center_region)
        channel_features.extend(stats_center)

        # Stats on upper region
        upper_region = channel_data[:channel_data.shape[0] // 2, :]
        stats_upper = calculate_stats(upper_region)
        channel_features.extend(stats_upper)


        features.extend(channel_features)

    return features # Ensure float list


# Helper function for calculating stats
def calculate_stats(data):
    """计算8个统计量, handles potential empty data"""
    data_flat = data.flatten()
    if data_flat.size == 0:
        # Return zeros or NaNs for empty regions, depending on desired behavior
        # Returning zeros might be safer for GP operations
        return [0.0] * 8

    mean_val = np.mean(data_flat)
    std_val = np.std(data_flat)

    # Handle potential std_val = 0 for kurtosis/skewness
    kurtosis_val = kurtosis(data_flat) if std_val > 1e-6 else 0.0
    skewness_val = skewness(data_flat) if std_val > 1e-6 else 0.0

    # Ensure data is non-negative for entropy log
    hist, _ = np.histogram(data, bins=256) # Using original shape data for histogram
    prob = hist / hist.sum()
    entropy_val = -np.sum(prob * np.log2(prob + 1e-7)) # Add epsilon for log(0)

    # Energy related stats
    energy_val = np.sum(data_flat ** 2)
    avg_energy_val = np.mean(data_flat ** 2)
    norm_val = np.linalg.norm(data_flat)


    return [
        float(mean_val),
        float(std_val),
        float(entropy_val),
        float(kurtosis_val),
        float(energy_val),
        float(avg_energy_val),
        float(norm_val),
        float(skewness_val)
    ]

def entropy(data):
    hist, _ = np.histogram(data, bins=256)
    prob = hist / (hist.sum() + 1e-7) # Add epsilon
    prob = prob[prob > 0] # Only consider non-zero probabilities
    return -np.sum(prob * np.log2(prob)) if prob.size > 0 else 0.0 # Handle case with no non-zero probs


def kurtosis(data):
     if len(data) < 4: return 0.0 # Kurtosis requires at least 4 data points
     # Standard kurtosis calculation
     mean = np.mean(data)
     std = np.std(data)
     if std < 1e-6: return 0.0 # Avoid division by zero
     return (np.mean((data - mean)**4) / (std**4)) - 3

def skewness(data):
     if len(data) < 3: return 0.0 # Skewness requires at least 3 data points
     # Standard skewness calculation
     mean = np.mean(data)
     std = np.std(data)
     if std < 1e-6: return 0.0 # Avoid division by zero
     return np.mean((data - mean)**3) / (std**3)


# --- Define Scalar GP Primitives ---
def add_s(a, b):
    return a + b

def sub_s(a, b):
    return a - b

def mul_s(a, b):
    return a * b

def protectedDiv_s(a, b):
    # Scalar protected division
    if b == 0:
        return 0.0 # Return 0 when divided by zero
    return a / b

def sin_s(a):
    return math.sin(a)

def cos_s(a):
    return math.cos(a)

def if_s(a, b, c, d):
    # Scalar conditional operator: if a < b return c, else return d
    return c if a < b else d

def get_feature_value_func(vector_input, index_value):
    """
    Function that takes the feature vector (list) and an index (int)
    and returns the float value at that index.
    Used by the GetValue primitive.
    """
    # These checks are safety measures, ideally types should be correct from Pset
    if not isinstance(vector_input, list):
         # print(f"Warning: get_feature_value_func received input of type {type(vector_input)}, expected list")
         # Attempt conversion or return default
         try: vector_input = list(vector_input)
         except: return 0.0
    if not isinstance(index_value, int):
         # print(f"Warning: get_feature_value_func received index of type {type(index_value)}, expected int")
         # Attempt conversion or return default
         try: index_value = int(index_value)
         except: return 0.0

    if 0 <= index_value < len(vector_input):
        return float(vector_input[index_value]) # Ensure float return type
    else:
        # Handle invalid index (shouldn't happen with correct Pset setup)
        # print(f"Warning: get_feature_value_func received invalid index {index_value} for vector length {len(vector_input)}")
        return 0.0
