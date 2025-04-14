import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import collections
from fgp_functions import uniform_lbp_gray, uniform_lbp_rgb, wavelet_transform


def load_images_from_folders(image_folders, metadata_path):
    metadata = pd.read_csv(metadata_path)
    class_mapping = {cls: idx for idx, cls in enumerate(metadata['dx'].unique())}

    combined_features = []
    labels = []
    loaded_images = 0
    expected_images = len(metadata)

    print("开始遍历文件夹")
    for folder in image_folders:
        if not os.path.exists(folder):
            print(f"警告：文件夹 {folder} 不存在")
            continue
        for image_name in os.listdir(folder):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(folder, image_name)
                image_id = os.path.splitext(image_name)[0]

                # 检查元数据中是否存在该 image_id
                if image_id not in metadata['image_id'].values:
                    print(f"警告：{image_id} 在元数据中不存在，跳过")
                    continue

                try:
                    # 加载图像，不调整大小
                    img_rgb = Image.open(image_path).convert('RGB')
                    img_array_uint8 = np.asarray(img_rgb, dtype=np.uint8)

                    print("开始提取特征")
                    # 提取特征
                    lgray_feat = uniform_lbp_gray(img_array_uint8)  # 59维
                    lrgb_feat = uniform_lbp_rgb(img_array_uint8)  # 177维
                    img_array_float = img_array_uint8.astype(np.float32) / 255.0

                    # 计算亮度通道并拼接4通道图像
                    luminance = (0.299 * img_array_float[:, :, 0] +
                                 0.587 * img_array_float[:, :, 1] +
                                 0.114 * img_array_float[:, :, 2])
                    luminance = np.expand_dims(luminance, axis=-1)
                    img_array_lum = np.concatenate([img_array_float, luminance], axis=-1)

                    wavelet_feat = wavelet_transform(img_array_lum)  # 416维

                    print("合并特征")
                    # 合并特征向量
                    merged_feature = np.concatenate([lgray_feat, lrgb_feat, wavelet_feat])
                    print(
                        f"特征维度-- LBP-Gray: {len(lgray_feat)}, "
                        f"LBP-RGB: {len(lrgb_feat)}, "
                        f"小波: {len(wavelet_feat)}, "
                        f"合并后: {len(merged_feature)}"
                    )
                    combined_features.append(merged_feature)

                    # 获取标签
                    cls = metadata[metadata['image_id'] == image_id]['dx'].values[0]
                    labels.append(class_mapping[cls])
                    loaded_images += 1

                except Exception as e:
                    print(f"错误：加载 {image_path} 失败，原因：{str(e)}")
                    continue

    print(f"预期图像数：{expected_images}，实际加载图像数：{loaded_images}")
    if loaded_images < expected_images:
        print(f"警告：仅加载了 {loaded_images}/{expected_images} 张图像")

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

def analyze_feature_ranges(x_data):
    n_samples, n_features = x_data.shape
    print(f"样本数：{n_samples}，特征数：{n_features}")

    # 分段提取
    lgray_range = (0, 59)        # 0-58
    lrgb_range = (59, 236)       # 59-235
    wavelet_range = (236, 652)   # 236-651

    ranges = {
        'LBP-Gray': x_data[:, lgray_range[0]:lgray_range[1]],
        'LBP-RGB': x_data[:, lrgb_range[0]:lrgb_range[1]],
        'Wavelet': x_data[:, wavelet_range[0]:wavelet_range[1]]
    }


def read_images(dataset_name, path):
    image_folders = [
        os.path.join(path, 'HAM10000_images_part_1'),
        os.path.join(path, 'HAM10000_images_part_2')
    ]
    metadata_path = os.path.join(path, 'HAM10000_metadata.csv')

    # 加载所有图像数据
    x_data, y_data = load_images_from_folders(image_folders, metadata_path)
    print("加载图像完成")

    # 分层划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data,
        train_size=0.8,
        stratify=y_data,
        random_state=42
    )
    print("划分数据集完成")

    # 只对训练集进行平衡处理
    x_train_balanced, y_train_balanced = balance_dataset(x_train, y_train)

    # 测试集保持不变，直接使用 x_test 和 y_test，不做平衡
    return x_train_balanced, x_test, y_train_balanced, y_test


if __name__ == "__main__":
    data_path = 'E:/datasets/archive'
    output_dir = os.path.join(data_path, 'processed_balanced_new')
    os.makedirs(output_dir, exist_ok=True)

    x_train, x_test, y_train, y_test = read_images('HAM10000', data_path)

    print("开始保存")
    # 保存处理后的数据
    np.save(os.path.join(output_dir, 'HAM10000_train_features.npy'), x_train)
    np.save(os.path.join(output_dir, 'HAM10000_test_features.npy'), x_test)
    np.save(os.path.join(output_dir, 'HAM10000_train_labels.npy'), y_train)
    np.save(os.path.join(output_dir, 'HAM10000_test_labels.npy'), y_test)
    print("保存成功")

    # 输出数据集属性
    prop = [
        f'Number of Classes {len(np.unique(y_train))}',
        f'Train Samples per Class: {len(y_train) // len(np.unique(y_train))}',
        f'Total Test Samples: {len(y_test)}',  # 测试集样本总数，不按类平衡
    ]
    with open(os.path.join(output_dir, 'HAM10000_properties.txt'), 'w') as f:
        f.write('\n'.join(prop))
        f.write(f'\nTrain Distribution {collections.Counter(y_train)}')
        f.write(f'\nTest Distribution {collections.Counter(y_test)}')

    # 验证数据形状
    print(f"Train features shape: {x_train.shape}")
    print(f"Test features shape: {x_test.shape}")

    print("数据处理完成")