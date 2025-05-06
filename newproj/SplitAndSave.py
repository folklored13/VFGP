import numpy as np
import os
import pandas as pd
from skimage import io
from skimage.transform import resize
import fgp_functions as fe_fs
from sklearn.model_selection import train_test_split # 导入划分工具
from tqdm import tqdm

# --- 路径 ---
IMAGE_BASE_FOLDER = "E:/datasets/PH2-dataset/images"
# IMAGE_BASE_FOLDER = "./PH2_Dataset_images/"
CSV_PATH = "E:/datasets/PH2-dataset/PH2_simple_dataset.csv"
PROCESSED_FOLDER_BASE = "E:/datasets/PH2-dataset/processed_ph2_separate_split/"
TEST_SIZE = 0.30
RANDOM_STATE = 42


CLASS_MAP = {
    'Common Nevus': 0,
    'Atypical Nevus': 1,
    'Melanoma': 2
}
# --- 配置结束 ---

def preprocess_and_extract(image_path):

    try:
        img = io.imread(image_path)
        if img.ndim == 2: img = np.stack((img,) * 3, axis=-1)
        if img.shape[2] == 4: img = img[:, :, :3]
        if img.dtype != np.uint8:
            if img.max() <= 1.0 and img.min() >= 0.0: img = (img * 255).astype(np.uint8)
            elif img.max() > 255 or img.min() < 0: img = np.clip(img, 0, 255).astype(np.uint8)
            else: img = img.astype(np.uint8)


        lgray_feat = fe_fs.uniform_lbp_gray(img)
        lrgb_feat = fe_fs.uniform_lbp_rgb(img)
        wavelet_feat = fe_fs.wavelet_transform(img)

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
    except Exception as e:
        print(f"Error reading CSV: {e}"); exit()

    if not os.path.isdir(IMAGE_BASE_FOLDER):
        print(f"Error: Image base folder not found: {IMAGE_BASE_FOLDER}"); exit()

    # 创建输出主目录
    os.makedirs(PROCESSED_FOLDER_BASE, exist_ok=True)
    output_paths = {
        "lbpgray": os.path.join(PROCESSED_FOLDER_BASE, "lbpgray"),
        "lbprgb": os.path.join(PROCESSED_FOLDER_BASE, "lbprgb"),
        "wavelet": os.path.join(PROCESSED_FOLDER_BASE, "wavelet")
    }
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    # --- 提取所有特征和标签 ---
    all_features_lbpgray = []
    all_features_lbprgb = []
    all_features_wavelet = []
    all_labels = []
    image_indices = [] # 记录成功处理的样本索引，用于后续划分

    print(f"Starting feature extraction...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Features"):
        image_name = row['image_name']
        diagnosis = row['diagnosis']

        base_name = os.path.splitext(image_name)[0]
        image_path = None
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
        for ext in possible_extensions:
            potential_path = os.path.join(IMAGE_BASE_FOLDER, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path is None or diagnosis not in CLASS_MAP:
            continue # 跳过找不到或标签无效的

        lgray, lrgb, wave = preprocess_and_extract(image_path)

        if lgray is not None and lrgb is not None and wave is not None:
            all_features_lbpgray.append(lgray)
            all_features_lbprgb.append(lrgb)
            all_features_wavelet.append(wave)
            all_labels.append(CLASS_MAP[diagnosis])
            image_indices.append(index) # 记录原始 DataFrame 行号或图片列表索引

    X_gray = np.array(all_features_lbpgray, dtype=np.float32)
    X_rgb = np.array(all_features_lbprgb, dtype=np.float32)
    X_wavelet = np.array(all_features_wavelet, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"\nTotal successfully processed images: {len(y)}")
    if len(y) == 0: print("Error: No data processed."); exit()

    # --- 执行 70/30 划分 ---

    print(f"\nSplitting data into {1-TEST_SIZE:.0%} train and {TEST_SIZE:.0%} test sets...")
    # 首先划分索引
    train_indices, test_indices = train_test_split(
        image_indices, # 使用成功处理的样本索引列表
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y # 根据标签进行分层抽样
    )


    map_original_idx_to_processed_idx = {orig_idx: proc_idx for proc_idx, orig_idx in enumerate(image_indices)}
    train_processed_indices = [map_original_idx_to_processed_idx[i] for i in train_indices]
    test_processed_indices = [map_original_idx_to_processed_idx[i] for i in test_indices]


    # 根据划分的索引切分数据
    y_train, y_test = y[train_processed_indices], y[test_processed_indices]

    feature_sets_all = {
        "lbpgray": X_gray,
        "lbprgb": X_rgb,
        "wavelet": X_wavelet
    }

    for name, X_all in feature_sets_all.items():
        output_dir = output_paths[name]
        print(f"\nProcessing and saving split for '{name}' to {output_dir}...")

        X_train, X_test = X_all[train_processed_indices], X_all[test_processed_indices]

        print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"  Train labels distribution: {np.bincount(y_train)}")
        print(f"  Test labels distribution: {np.bincount(y_test)}")

        # 保存训练集和测试集
        np.save(os.path.join(output_dir, f"train_features_{name}.npy"), X_train)
        np.save(os.path.join(output_dir, f"train_labels_{name}.npy"), y_train)
        np.save(os.path.join(output_dir, f"test_features_{name}.npy"), X_test)
        np.save(os.path.join(output_dir, f"test_labels_{name}.npy"), y_test)

    print("\nData preparation and splitting finished.")