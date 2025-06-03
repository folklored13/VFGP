import numpy as np
import os
data_path = "E:/datasets/PH2-dataset/processed_ph2_runs/lbpgray/run9/"
x_train = np.load(os.path.join(data_path, "train_features_lbpgray.npy"))
y_train = np.load(os.path.join(data_path, "train_labels_lbpgray.npy"))
print(x_train)
print(y_train)