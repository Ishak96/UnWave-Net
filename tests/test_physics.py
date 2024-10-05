from matplotlib import pyplot as plt

import sys
sys.path.append("../")

from src.pl_modules import CSTDataModule
from src.physics import *

from src.util import to_hu

size = 256
num_detectors = 150
poission_level = 1e6
gaussian_level = None

json_path = "../src/json/mayo.json"
matrix_path = f"../radon/{num_detectors}"

# create the physics module
cst = NCCCST(matrix_path=matrix_path, size=size, num_detectors=num_detectors, emax=100)

# create data module
mayodata = CSTDataModule(json_path=json_path, size=size, num_detectors=num_detectors, emax=100, poission_level=poission_level, gaussian_level=gaussian_level)
mayodata.setup(stage="test")

# retrive a batch of data
dataloader = mayodata.test_dataloader()
sample = dataloader.dataset[21]

y = sample.g_data
target = sample.target
min_max_value = sample.min_max_value

# apply the physics model
x_out = cst.adjoint_operator(y)
print(f"Shape of the output: {x_out.shape}")

target_hu = to_hu(target, min_max_value, sample.hu_information, -1000., 800., False)
x_out_hu = to_hu(x_out, min_max_value, sample.hu_information, -1000., 800., False)

# Plot the target and the g_data
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(target_hu[0, ...], cmap="gray")
plt.title("Target")
plt.subplot(1, 3, 2)
plt.imshow(y[0, ...], cmap="gray")
plt.title("g_data")
plt.subplot(1, 3, 3)
plt.imshow(x_out_hu[0, ...].detach().numpy(), cmap="gray")
plt.title("x_out")
plt.show()