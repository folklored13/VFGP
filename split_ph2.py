import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
import collections
from fgp_functions import uniform_lbp_gray, uniform_lbp_rgb, wavelet_transform


def load_images_from_folders(image_folder, metadata_path):
    combined_features = []
    labels = []

    # 读取元数据
    metadata = pd.read_csv(metadata_path)
    class_mapping = {"Common Nevus": 0, "Atypical Nevus": 1, "Melanoma": 2}

    for _, row in metadata.iterrows():
        image_name = row["image_name"]
        diagnosis = row["diagnosis"]

        # 构建图像路径
        image_path = os.path.join(image_folder, f"{image_name}.jpg")

        if not os.path.exists(image_path):
            print(f"警告：{image_path} 不存在，跳过")
            continue

        try:
            # 加载图像并提取特征
            img_rgb = Image.open(image_path).convert('RGB')
            img_array_uint8 = np.asarray(img_rgb)

            # 特征提取（保持原有逻辑）
            lgray_feat = uniform_lbp_gray(img_array_uint8)
            lrgb_feat = uniform_lbp_rgb(img_array_uint8)

            # 亮度通道计算
            img_array_float = img_array_uint8.astype(np.float32) / 255.0
            luminance = 0.299 * img_array_float[..., 0] + 0.587 * img_array_float[..., 1] + 0.114 * img_array_float[
                ..., 2]
            luminance = np.expand_dims(luminance, axis=-1)
            img_array_lum = np.concatenate([img_array_float, luminance], axis=-1)

            wavelet_feat = wavelet_transform(img_array_lum)

            # 合并特征
            merged_feature = np.concatenate([lgray_feat, lrgb_feat, wavelet_feat])
            combined_features.append(merged_feature)
            labels.append(class_mapping[diagnosis])

        except Exception as e:
            print(f"错误处理 {image_path}: {str(e)}")

    print(
        f"特征维度-- LBP-Gray: {len(lgray_feat)}, "
        f"LBP-RGB: {len(lrgb_feat)}, "
        f"小波: {len(wavelet_feat)}, "
        f"合并后: {len(merged_feature)}"
    )
    return np.array(combined_features), np.array(labels)

def balance_dataset(x_data, y_data):
    """返回平衡后的数据，仅用于训练集"""
    unique_classes, class_counts = np.unique(y_data, return_counts=True)
    min_samples = min(class_counts)

    balanced_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_data == cls)[0]
        np.random.shuffle(cls_indices)
        balanced_indices.extend(cls_indices[:min_samples])

    np.random.shuffle(balanced_indices)
    return x_data[balanced_indices], y_data[balanced_indices]


def read_images(dataset_name, path, use_kfold=False):
    # 路径配置
    image_folder = os.path.join(path, "images")
    metadata_path = os.path.join(path, "PH2_simple_dataset.csv")

    # 加载数据
    x_data, y_data = load_images_from_folders(image_folder, metadata_path)

    # 数据划分
    if use_kfold:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_data = []
        for train_idx, test_idx in skf.split(x_data, y_data):
            x_train, x_test = x_data[train_idx], x_data[test_idx]
            y_train, y_test = y_data[train_idx], y_data[test_idx]
            fold_data.append((x_train, x_test, y_train, y_test))
        return fold_data
    else:
        return train_test_split(
            x_data, y_data,
            test_size=0.2,
            stratify=y_data,
            random_state=42
        )


if __name__ == "__main__":
    data_path = "E:/datasets/PH2-dataset"
    output_dir = os.path.join(data_path, "processed_ph2")
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    x_train, x_test, y_train, y_test = read_images("PH2", data_path)

    # 保存处理后的数据
    np.save(os.path.join(output_dir, "train_features.npy"), x_train)
    np.save(os.path.join(output_dir, "test_features.npy"), x_test)
    np.save(os.path.join(output_dir, "train_labels.npy"), y_train)
    np.save(os.path.join(output_dir, "test_labels.npy"), y_test)


    # 输出数据集属性
    prop_path = os.path.join(output_dir, 'ph2_properties.txt')
    with open(prop_path, 'w') as f:
        f.write(f"训练集形状: {x_train.shape}\n")
        f.write(f"测试集形状: {x_test.shape}\n\n")

        # 训练集类别分布
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        f.write("训练集类别分布:\n")
        for cls, count in zip(train_unique, train_counts):
            f.write(f"  类别 {cls}: {count} 样本\n")

        # 测试集类别分布
        test_unique, test_counts = np.unique(y_test, return_counts=True)
        f.write("\n测试集类别分布:\n")
        for cls, count in zip(test_unique, test_counts):
            f.write(f"  类别 {cls}: {count} 样本\n")

    print(f"属性保存至：{prop_path}")

    print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")
    print(f"类别分布 - 训练集: {np.unique(y_train, return_counts=True)}")
    print(f"类别分布 - 测试集: {np.unique(y_test, return_counts=True)}")


    print("数据处理完成")