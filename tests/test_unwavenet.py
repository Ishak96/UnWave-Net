from matplotlib import pyplot as plt

import sys
sys.path.append("../")

from src.pl_modules import CSTDataModule
from src.pl_modules import *
from src.physics import NCCCST

from src.util import to_hu

size = 256
num_detectors = 150
emax = 100
poission_level = 1e6
gaussian_level = None

json_path = "../src/json/mayo.json"
matrix_path = f"../radon/{num_detectors}"

# create the physics module
cst = NCCCST(matrix_path=matrix_path, size=size, num_detectors=num_detectors, emax=emax)

# create data model
model = UnWaveNetModule(physics_model=cst, num_cascades=5).cuda()

# create data module
mayodata = CSTDataModule(json_path=json_path, size=size, num_detectors=num_detectors, emax=emax, poission_level=poission_level, gaussian_level=gaussian_level)
mayodata.setup(stage="test")

# retrive a batch of data
dataloader = mayodata.test_dataloader()
sample = dataloader.dataset[21]

y = sample.g_data
target = sample.target
min_max_value = sample.min_max_value

# apply the model
x_out = model(y.cuda())
print(f"Shape of the output: {x_out.shape}")

target_hu = to_hu(target, min_max_value, sample.hu_information, -1000., 800., False)
x_out_hu = to_hu(x_out, min_max_value, sample.hu_information, -1000., 800., False)
x_out_hu = x_out_hu.cpu()

# Plot the target and the g_data
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(target_hu[0, ...], cmap="gray")
plt.title("Target")
plt.subplot(1, 3, 2)
plt.imshow(y[0, ...], cmap="gray")
plt.title("g_data")
plt.subplot(1, 3, 3)
plt.imshow(x_out_hu[0, 0, ...].detach().numpy(), cmap="gray")
plt.title("x_out")
plt.show()