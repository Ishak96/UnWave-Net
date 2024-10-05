from matplotlib import pyplot as plt
import numpy as np
import torch

import sys
sys.path.append("../")

from src.pl_modules import CSTDataModule
from src.util import to_hu

size = 256
num_detectors = 150
emax = 100
poission_level = 1e6
gaussian_level = 0.05

json_path = "../src/json/mayo.json"

# read the radon matrix
A = torch.load(f"../radon/{num_detectors}/A_256x256.pt")
AT = torch.load(f"../radon/{num_detectors}/AT_256x256.pt")
print(f"Radon matrix shape: {A.shape}")
print(f"Pseudo inverse shape: {AT.shape}")

# create data module
mayodata = CSTDataModule(json_path=json_path, size=size, num_detectors=num_detectors, emax=emax, add_noise=True, poission_level=poission_level, gaussian_level=gaussian_level)
mayodata.setup(stage="test")

# retrive a batch of data
dataloader = mayodata.test_dataloader()
sample = dataloader.dataset[21]

g_data = sample.g_data
target = sample.target
A_g_data = torch.sparse.mm(A, target.reshape(size ** 2, 1)).reshape(g_data.shape)
A_x = torch.mm(AT, g_data.reshape(g_data.shape[1] * g_data.shape[2], 1)).reshape(1, size, size)

max_value = sample.max_value
file_name = sample.file_name
min_max_value = sample.min_max_value
hu_information = sample.hu_information

# Print all the mix max values
print(f"Target: {target.min()}, {target.max()}")

print(f"Dicom information: {sample.hu_information}")
print(f"Min max value: {min_max_value}")
print(f"Filename: {file_name}")

target_hu = to_hu(target, min_max_value, sample.hu_information, -1000., 800., False)

# Plot the target and the g_data
plt.figure(figsize=(10, 10))
plt.subplot(1, 4, 1)
plt.imshow(target_hu[0, ...], cmap="gray")
plt.title("Target")
plt.subplot(1, 4, 2)
plt.imshow(g_data[0, ...], cmap="gray")
plt.title("g_data")
plt.subplot(1, 4, 3)
plt.imshow(A_g_data[0, ...], cmap="gray")
plt.title("A * target")
plt.subplot(1, 4, 4)
plt.imshow(A_x[0, ...], cmap="gray")
plt.title("A^T * g_data")
plt.show()