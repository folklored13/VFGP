import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import collections

from fgp_functions import uniform_lbp_gray, uniform_lbp_rgb, wavelet_transform


def load_images_from_folders(image_folders, metadata_path, width, height):
    metadata = pd.read_csv(metadata_path)
    class_mapping = {cls: idx for idx, cls in enumerate(metadata['dx'].unique())}

    # imgs_rgb = []
    # imgs_luminance = []
    combined_features = []
    labels = []

    for folder in image_folders:
        for image_name in os.listdir(folder):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(folder, image_name)

                img_rgb = Image.open(image_path).convert('RGB')
                img_rgb = img_rgb.resize((width, height), Image.BILINEAR)
                img_array_uint8 = np.asarray(img_rgb, dtype=np.uint8)  # 先转为uint8
                lgray_feat = uniform_lbp_gray(img_array_uint8)  # 59维
                lrgb_feat = uniform_lbp_rgb(img_array_uint8)  # 177维
                img_array_float = img_array_uint8.astype(np.float32) / 255.0

                #计算亮度通道拼接4通道图像
                luminance = 0.299 * img_array_float[:, :, 0] + 0.587 * img_array_float[:, :, 1] + 0.114 * img_array_float[:,
                                                                                                      :, 2]
                luminance = np.expand_dims(luminance, axis=-1)
                img_array_lum = np.concatenate([img_array_float, luminance], axis=-1)

                #提取三种特征
                # imgs_rgb.append(img_array_rgb)
                # imgs_luminance.append(img_array_with_lum)

                wavelet_feat = wavelet_transform(img_array_lum)  # 416维

                #合并特征向量
                merged_feature = np.concatenate([lgray_feat, lrgb_feat, wavelet_feat])
                #assert len(merged_feature) == 652, "特征维度错误"
                print(
                    f"特征维度-- LBP-Gray: {len(lgray_feat)}, "
                    f"LBP-RGB: {len(lrgb_feat)}, "
                    f"小波: {len(wavelet_feat)}, "
                    f"合并后: {len(merged_feature)}"
                )
                combined_features.append(merged_feature)

                #获取标签
                image_id = os.path.splitext(image_name)[0]
                cls = metadata[metadata['image_id'] == image_id]['dx'].values[0]
                labels.append(class_mapping[cls])

    return np.array(combined_features), np.array(labels)


def balance_dataset(x_data, y_data):
    """返回平衡后的数据和对应的索引"""
    unique_classes, class_counts = np.unique(y_data, return_counts=True)
    min_samples = min(class_counts)

    balanced_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_data == cls)[0]
        np.random.shuffle(cls_indices)
        balanced_indices.extend(cls_indices[:min_samples])

    np.random.shuffle(balanced_indices)
    return x_data[balanced_indices], y_data[balanced_indices]


def read_images(dataset_name, path, width, height):
    image_folders = [
        os.path.join(path, 'HAM10000_images_part_1'),
        os.path.join(path, 'HAM10000_images_part_2')
    ]
    metadata_path = os.path.join(path, 'HAM10000_metadata.csv')

    #x_data_rgb, x_data_lum, y_data = load_images_from_folders(image_folders, metadata_path, width, height)
    x_data, y_data = load_images_from_folders(image_folders, metadata_path, width, height)

    # 分层划分训练集和测试集（8:2）
    # x_train_rgb, x_test_rgb, x_train_lum, x_test_lum, y_train, y_test = train_test_split(
    #     x_data_rgb, x_data_lum, y_data,
    #     train_size=0.8, stratify=y_data, random_state=12
    # )
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data,
        train_size=0.8,
        stratify=y_data,
        random_state=42
    )

    # 分别平衡训练集和测试集，并获取索引
    # x_train_rgb_bal, y_train_bal, train_indices = balance_dataset(x_train_rgb, y_train)
    # x_train_lum_bal = x_train_lum[train_indices]  # 直接使用平衡后的索引
    # x_test_rgb_bal, y_test_bal, test_indices = balance_dataset(x_test_rgb, y_test)
    # x_test_lum_bal = x_test_lum[test_indices]  # 直接使用平衡后的索引

    x_train,y_train = balance_dataset(x_train,y_train)
    x_test,y_test = balance_dataset(x_test,y_test)

    #return x_train_rgb_bal, x_train_lum_bal, x_test_rgb_bal, x_test_lum_bal, y_train_bal, y_test_bal
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    data_path = 'E:/datasets/archive'
    output_dir = os.path.join(data_path, 'processed_balanced')
    os.makedirs(output_dir, exist_ok=True)

    wid, hei = 64, 64

    x_train, x_test, y_train, y_test = read_images('HAM10000', data_path, wid, hei)

    np.save(os.path.join(output_dir, 'HAM10000_train_features.npy'), x_train)
    np.save(os.path.join(output_dir, 'HAM10000_test_features.npy'), x_test)
    np.save(os.path.join(output_dir, 'HAM10000_train_labels.npy'), y_train)
    np.save(os.path.join(output_dir, 'HAM10000_test_labels.npy'), y_test)

    # np.save(os.path.join(output_dir, 'HAM10000_train_rgb.npy'), x_train_rgb)
    # np.save(os.path.join(output_dir, 'HAM10000_train_lum.npy'), x_train_lum)
    # np.save(os.path.join(output_dir, 'HAM10000_test_rgb.npy'), x_test_rgb)
    # np.save(os.path.join(output_dir, 'HAM10000_test_lum.npy'), x_test_lum)
    # np.save(os.path.join(output_dir, 'HAM10000_train_label.npy'), y_train)
    # np.save(os.path.join(output_dir, 'HAM10000_test_label.npy'), y_test)

    prop = [
        f'Number of Classes {len(np.unique(y_train))}',
        f'Train Samples per Class: {len(y_train) // len(np.unique(y_train))}',
        f'Test Samples per Class: {len(y_test) // len(np.unique(y_test))}',
        #f'Image Size {x_train_rgb.shape[1]} {x_train_rgb.shape[2]}'
    ]
    with open(os.path.join(output_dir, 'HAM10000_properties.txt'), 'w') as f:
        f.write('\n'.join(prop))
        f.write(f'\nTrain Distribution {collections.Counter(y_train)}')
        f.write(f'\nTest Distribution {collections.Counter(y_test)}')


    #print("Combined feature shape:", combined_features[0].shape)  # 应为 (652,)
    # 验证数据形状
    print(f"Train features shape: {x_train.shape}")  # 应输出 (N_train, 652)
    print(f"Test features shape: {x_test.shape}")  # 应输出 (N_test, 652)