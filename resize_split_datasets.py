import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import collections


def load_images_from_folders(image_folders, metadata_path, width, height):
    #读取metadata
    metadata = pd.read_csv(metadata_path)
    class_mapping = {cls: idx for idx, cls in enumerate(metadata['dx'].unique())}

    imgs_rgb = []  # 存储RGB三通道图像
    imgs_luminance = []  # 存储亮度通道图像
    labels = []


    for folder in image_folders:
        for image_name in os.listdir(folder):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(folder, image_name)
                #加载RGB图像
                img_rgb = Image.open(image_path).convert('RGB')  #保留RGB三通道
                img_rgb = img_rgb.resize((width, height), Image.BILINEAR)
                img_array_rgb = np.asarray(img_rgb, dtype='float32')  #形状：(H, W, 3)

                #生成亮度通道Luminance
                luminance = 0.299 * img_array_rgb[:, :, 0] + 0.587 * img_array_rgb[:, :, 1] + 0.114 * img_array_rgb[:,:, 2]
                luminance = np.expand_dims(luminance, axis=-1)  #形状：(H, W, 1)

                #合并RGB和亮度通道，小波特征提取
                img_array_with_lum = np.concatenate([img_array_rgb, luminance], axis=-1)  #形状：(H, W, 4)

                imgs_rgb.append(img_array_rgb)
                imgs_luminance.append(img_array_with_lum)

                #从元数据中获取类别0-6
                image_id = os.path.splitext(image_name)[0]
                cls = metadata[metadata['image_id'] == image_id]['dx'].values[0]
                labels.append(class_mapping[cls])

    #返回rgb和带亮度通道的图像
    return np.array(imgs_rgb), np.array(imgs_luminance), np.array(labels)


def read_images(dataset_name, num_instance, path, width, height):
    #图像路径
    image_folders = [
        os.path.join(path, 'HAM10000_images_part_1'),
        os.path.join(path, 'HAM10000_images_part_2')
    ]
    metadata_path = os.path.join(path, 'HAM10000_metadata.csv')

    #加载图像和标签
    x_data_rgb, x_data_lum, y_data = load_images_from_folders(image_folders, metadata_path, width, height)

    #划分训练集和测试集8:2 取消训练集固定数量 调整比例，设train_size参数为0.8
    x_train_rgb, x_test_rgb, x_train_lum, x_test_lum, y_train, y_test = train_test_split(
        x_data_rgb, x_data_lum, y_data, train_size=0.8, random_state=12)

    return x_train_rgb,x_train_lum,x_test_rgb,x_test_lum, y_train, y_test


if __name__ == "__main__":
    data_path = 'E:/datasets/archive'
    output_dir = os.path.join(data_path, 'processed')
    os.makedirs(output_dir, exist_ok=True)

    wid = 32
    hei = 32
    num_ins = 10  #训练集样本量

    x_train_rgb,x_train_lum,x_test_rgb,x_test_lum, y_train, y_test = read_images('HAM10000', num_ins, data_path, wid, hei)

    #保存数据
    np.save(os.path.join(output_dir, 'HAM10000_train_rgb.npy'), x_train_rgb)
    np.save(os.path.join(output_dir, 'HAM10000_train_lum.npy'), x_train_lum)
    np.save(os.path.join(output_dir, 'HAM10000_test_rgb.npy'), x_test_rgb)
    np.save(os.path.join(output_dir, 'HAM10000_test_lum.npy'), x_test_lum)
    np.save(os.path.join(output_dir, 'HAM10000_train_label.npy'), y_train)
    np.save(os.path.join(output_dir, 'HAM10000_test_label.npy'), y_test)

    #保存属性文件
    prop = [
        f'Number of Classes {len(np.unique(y_train))}',
        f'Instances for training {len(x_train_rgb)}',
        f'Instances for training {len(x_train_lum)}',
        f'Instances for testing {len(x_test_rgb)}',
        f'Instances for testing {len(x_test_lum)}',
        f'Image Size {x_train_rgb.shape[1]} {x_train_rgb.shape[2]}'
    ]
    with open(os.path.join(output_dir, 'HAM10000_properties.txt'), 'w') as f:
        f.write('\n'.join(prop))
        f.write(f'\nTrain {collections.Counter(y_train)}')
        f.write(f'\nTest {collections.Counter(y_test)}')