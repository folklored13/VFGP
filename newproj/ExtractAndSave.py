import numpy as np
import os
import pandas as pd
from skimage import io
from skimage.transform import resize
import fgp_functions as fe_fs
from tqdm import tqdm # 用于显示进度条: pip install tqdm

# --- 配置 ---
# 原始 PH2 图像文件夹路径 (包含 IMDXXX.jpg 文件)

IMAGE_BASE_FOLDER = "E:/datasets/PH2-dataset/images"

# IMAGE_BASE_FOLDER = "./PH2_Dataset_images/"

# CSV 文件路径
CSV_PATH = "E:/datasets/PH2-dataset/PH2_simple_dataset.csv"

# 处理后特征保存的基础路径
PROCESSED_FOLDER_BASE = "E:/datasets/PH2-dataset/processed_ph2_separate/"

# 类别到数字标签的映射
# !!! 确保这个顺序与你的分类器期望的一致 !!!
CLASS_MAP = {
    'Common Nevus': 0,
    'Atypical Nevus': 1,
    'Melanoma': 2
}
# --- 配置结束 ---

def preprocess_and_extract(image_path):
    """加载、预处理图像并提取所有特征类型"""
    try:
        img = io.imread(image_path)
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # 检查并转换为 uint8, 0-255
        if img.dtype != np.uint8:
            if img.max() <= 1.0 and img.min() >= 0.0:
                img = (img * 255).astype(np.uint8)
            elif img.max() > 255 or img.min() < 0:
                 print(f"Warning: Image {os.path.basename(image_path)} has unusual range [{img.min()}, {img.max()}]. Clamping and converting.")
                 img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                 img = img.astype(np.uint8)

        # --- 可选：调整图像大小 ---
        # 如果原始图像大小不一，或者为了加速特征提取，可以取消注释下一行
        # 如果调整大小，确保所有图像调整到相同尺寸，例如 256x256
        # target_size = (256, 256)
        # if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
        #     # print(f"Resizing {os.path.basename(image_path)} from {img.shape[:2]} to {target_size}")
        #     img = resize(img, target_size, anti_aliasing=True, preserve_range=True).astype(np.uint8)
        # --- 调整大小结束 ---


        # 提取特征
        lgray_feat = fe_fs.uniform_lbp_gray(img) # 59 dim
        lrgb_feat = fe_fs.uniform_lbp_rgb(img)  # 177 dim
        wavelet_feat = fe_fs.wavelet_transform(img) # 416 dim

        # 维度检查
        if len(lgray_feat) != 59: print(f"Warning: LBP Gray dim mismatch for {os.path.basename(image_path)}")
        if len(lrgb_feat) != 177: print(f"Warning: LBP RGB dim mismatch for {os.path.basename(image_path)}")
        if len(wavelet_feat) != 416: print(f"Warning: Wavelet dim mismatch for {os.path.basename(image_path)}")

        return lgray_feat, lrgb_feat, wavelet_feat

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None, None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None


# --- 主逻辑 ---
if __name__ == "__main__":
    print(f"Reading dataset definition from: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Found {len(df)} images listed in CSV.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_PATH}")
        exit()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        exit()

    # 确保图像基础文件夹存在
    if not os.path.isdir(IMAGE_BASE_FOLDER):
         print(f"Error: Image base folder not found at {IMAGE_BASE_FOLDER}")
         exit()

    # 创建输出目录结构
    output_paths = {
        "lbpgray": os.path.join(PROCESSED_FOLDER_BASE, "lbpgray"),
        "lbprgb": os.path.join(PROCESSED_FOLDER_BASE, "lbprgb"),
        "wavelet": os.path.join(PROCESSED_FOLDER_BASE, "wavelet")
    }
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    all_features_lbpgray = []
    all_features_lbprgb = []
    all_features_wavelet = []
    all_labels = []
    processed_count = 0
    error_count = 0

    print(f"Starting feature extraction from images in: {IMAGE_BASE_FOLDER}")
    # 用tqdm 显示进度
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Images"):
        image_name = row['image_name']
        diagnosis = row['diagnosis']


        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
        image_path = None
        # 移除可能的 .jpg 后缀（如果 CSV 中包含）
        base_name = os.path.splitext(image_name)[0]
        for ext in possible_extensions:
            potential_path = os.path.join(IMAGE_BASE_FOLDER, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path is None:
            print(f"Warning: Could not find image file for {image_name} in {IMAGE_BASE_FOLDER}")
            error_count += 1
            continue

        if diagnosis not in CLASS_MAP:
            print(f"Warning: Unknown diagnosis '{diagnosis}' for image {image_name}. Skipping.")
            error_count += 1
            continue

        # 提取特征
        lgray_feat, lrgb_feat, wavelet_feat = preprocess_and_extract(image_path)

        if lgray_feat is not None and lrgb_feat is not None and wavelet_feat is not None:
            all_features_lbpgray.append(lgray_feat)
            all_features_lbprgb.append(lrgb_feat)
            all_features_wavelet.append(wavelet_feat)
            all_labels.append(CLASS_MAP[diagnosis])
            processed_count += 1
        else:
            error_count += 1

    print(f"\nFinished processing. Successfully processed: {processed_count}, Errors/Skipped: {error_count}")

    if processed_count == 0:
        print("Error: No images were processed successfully. Check paths and image formats.")
        exit()

    # 转换为 NumPy 数组
    X_gray = np.array(all_features_lbpgray, dtype=np.float32)
    X_rgb = np.array(all_features_lbprgb, dtype=np.float32)
    X_wavelet = np.array(all_features_wavelet, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32) # 使用整数类型

    print(f"\nFinal data shapes - Gray: {X_gray.shape}, RGB: {X_rgb.shape}, Wavelet: {X_wavelet.shape}, Labels: {y.shape}")

    # --- 分别保存每个特征集 ---
    feature_sets = {
        "lbpgray": X_gray,
        "lbprgb": X_rgb,
        "wavelet": X_wavelet
    }

    for name, data in feature_sets.items():
        output_dir = output_paths[name]
        print(f"\nSaving features for '{name}' to {output_dir}...")
        np.save(os.path.join(output_dir, f"all_features_{name}.npy"), data)
        np.save(os.path.join(output_dir, f"all_labels_{name}.npy"), y) # 标签对所有集都相同

    print("\nData preparation finished. Features saved separately based on CSV.")