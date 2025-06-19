import numpy as np
import os
import pandas as pd
from skimage import io
from skimage.transform import resize
import fgp_functions as fe_fs
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config

IMAGE_BASE_FOLDER = "E:/datasets/PH2-dataset/images"
# IMAGE_BASE_FOLDER = "./PH2_Dataset_images/"
CSV_PATH = "E:/datasets/PH2-dataset/PH2_simple_dataset.csv"
# 输出目录 ./processed_ph2_runs/lbpgray/run0/, ./processed_ph2_runs/lbpgray/run1/, ...
PROCESSED_RUNS_BASE = "E:/datasets/PH2-dataset/processed_ph2_runs/"
NUM_RUNS = 10
TEST_SIZE = 0.30

RANDOM_STATE_BASE = 42

CLASS_MAP = config.CLASS_MAP



def preprocess_and_extract(image_path):
    # ...
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



        return lgray_feat, lrgb_feat, wavelet_feat
    except FileNotFoundError:
        # print(f"Error: Image file not found at {image_path}")
        return None, None, None
    except Exception as e:
        # print(f"Error processing image {image_path}: {e}")
        return None, None, None



if __name__ == "__main__":
    print(f"Reading dataset definition from: {CSV_PATH}")
    try: df = pd.read_csv(CSV_PATH)
    except Exception as e: print(f"Error reading CSV: {e}"); exit()
    if not os.path.isdir(IMAGE_BASE_FOLDER): print(f"Error: Image base folder not found: {IMAGE_BASE_FOLDER}"); exit()

    os.makedirs(PROCESSED_RUNS_BASE, exist_ok=True)


    all_features_lbpgray = []
    all_features_lbprgb = []
    all_features_wavelet = []
    all_labels = []
    image_indices_for_split = [] # 用于记录处理的样本的索引

    print(f"Starting feature extraction for all images...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Features"):
        image_name = row['image_name']
        diagnosis = row['diagnosis']
        base_name = os.path.splitext(image_name)[0]
        image_path = None
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in possible_extensions:
            potential_path = os.path.join(IMAGE_BASE_FOLDER, base_name + ext)
            if os.path.exists(potential_path): image_path = potential_path; break
        if image_path is None or diagnosis not in CLASS_MAP: continue

        lgray, lrgb, wave = preprocess_and_extract(image_path)

        if lgray is not None and lrgb is not None and wave is not None:
            all_features_lbpgray.append(lgray)
            all_features_lbprgb.append(lrgb)
            all_features_wavelet.append(wave)
            all_labels.append(CLASS_MAP[diagnosis])
            image_indices_for_split.append(len(all_labels) - 1) # 添加新样本的索引

    X_gray_all = np.array(all_features_lbpgray, dtype=np.float32)
    X_rgb_all = np.array(all_features_lbprgb, dtype=np.float32)
    X_wavelet_all = np.array(all_features_wavelet, dtype=np.float32)
    y_all = np.array(all_labels, dtype=np.int32)

    print(f"\nTotal successfully processed images for splitting: {len(y_all)}")
    if len(y_all) == 0: print("Error: No data processed."); exit()

    # --- 执行 NUM_RUNS 次独立的 70/30 划分并保存 ---
    feature_sets_all_data = {
        "lbpgray": X_gray_all,
        "lbprgb": X_rgb_all,
        "wavelet": X_wavelet_all
    }

    print(f"\nCreating and saving {NUM_RUNS} independent 70/30 splits...")
    for run_idx in range(NUM_RUNS):
        # 使用不同的随机种子为每次运行创建不同的划分
        current_random_state = RANDOM_STATE_BASE + run_idx
        train_indices, test_indices = train_test_split(
            np.arange(len(y_all)), # 划分 y_all 的索引
            test_size=TEST_SIZE,
            random_state=current_random_state,
            stratify=None # 原来是stratify=y_all
        )

        y_train_run = y_all[train_indices]
        y_test_run = y_all[test_indices]

        print(f"  Processing Run {run_idx} (Random State: {current_random_state})")

        for feature_name, X_full_set in feature_sets_all_data.items():
            run_feature_dir = os.path.join(PROCESSED_RUNS_BASE, feature_name, f"run{run_idx}")
            os.makedirs(run_feature_dir, exist_ok=True)

            X_train_run = X_full_set[train_indices]
            X_test_run = X_full_set[test_indices]


            np.save(os.path.join(run_feature_dir, f"train_features_{feature_name}.npy"), X_train_run)
            np.save(os.path.join(run_feature_dir, f"train_labels_{feature_name}.npy"), y_train_run)
            np.save(os.path.join(run_feature_dir, f"test_features_{feature_name}.npy"), X_test_run)
            np.save(os.path.join(run_feature_dir, f"test_labels_{feature_name}.npy"), y_test_run)

    print(f"\nData preparation finished. {NUM_RUNS} runs of 70/30 splits saved for each feature set in '{PROCESSED_RUNS_BASE}'")