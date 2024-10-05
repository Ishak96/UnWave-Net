from typing import NamedTuple

import os

import pathlib

import torch
import numpy as np

import json
import pydicom
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T

from src.util import Normalization, HuInformation, MinMaxValue, get_hu_information

class MayoSample(NamedTuple):
    g_data: torch.Tensor
    target: torch.Tensor
    max_value: float
    file_name: str
    min_max_value: MinMaxValue = None
    hu_information: HuInformation = None

class MayoData(Dataset):
    def __init__(self,
                 json_path: str = None,
                 mode: str = 'train',
                 size: int = 256,
                 num_detectors: int = 100,
                 emax: int = 100,
                 add_noise: bool = False,
                 poission_level: float = 5e6,
                 gaussian_level: float = 0.05,
                 **kwargs,
    ):
        super(MayoData, self).__init__()

        self.num_detectors = num_detectors
        self.emax = emax
        self.add_noise = add_noise
        self.poission_level = poission_level
        self.gaussian_level = gaussian_level

        if mode not in ['train', 'val', 'test']:
            raise ValueError('Invalid mode: %s' % mode)
        
        self.img_transform = T.Compose([T.Resize(size), T.ToTensor(), Normalization()])

        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
            self.list_data_file = json_data[mode]

    def get_first_of_dicom_field_as_int(self, x):
        #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
        if type(x) == pydicom.multival.MultiValue: return int(x[0])
        else: return int(x)

    def __len__(self):
        return len(self.list_data_file)

    def __getitem__(self, index: int):
        for root, dirs, files in os.walk(self.list_data_file[index]['filename']):
            for file in files:
                if file.endswith('.IMA'):
                    img_name = os.path.join(root, file)
                elif file.endswith('.csv') and root.endswith(str(self.num_detectors)):
                    data_name = os.path.join(root, file)

        img_path = pathlib.Path(img_name)
        data_path = pathlib.Path(data_name)
        
        file_name = img_path.name

        img_data = pydicom.read_file(img_path)

        # get hu information
        hu_information = get_hu_information(img_data)

        target = img_data.pixel_array
        target = Image.fromarray(target.astype('float32'), mode='F')

        min_max_value = MinMaxValue(min_value=np.min(target), max_value=np.max(target))

        target = self.img_transform(target)
        max_value = torch.max(target)

        # get sinogram data by reading the csv file
        g_data = np.genfromtxt(data_path, delimiter=',')
        g_data = torch.from_numpy(g_data).float().T.unsqueeze(0)

        # the following part code is used to randomly choose sinograms to satisfy the sparse-view requeirement
        if self.add_noise:
            # add poission noise
            intensityI0 = self.poission_level
            scale_value = torch.from_numpy(np.array(intensityI0).astype(np.float32))
            normalized_sino = torch.exp(-g_data / g_data.max())
        
            th_data = np.random.poisson(scale_value * normalized_sino)
            sino_noisy = -torch.log(torch.from_numpy(th_data) / scale_value)
            g_data = sino_noisy * g_data.max()

            # add Gaussian noise
            if self.gaussian_level is not None:
                noise_std = self.gaussian_level
                noise_std = np.array(noise_std).astype(np.float32)
                nx, ny = np.array(self.num_detectors).astype(np.int32), np.array(self.emax).astype(np.int32)
                noise = noise_std * np.random.randn(nx, ny)
                noise = torch.from_numpy(noise)
                g_data = g_data + noise
                g_data = g_data.float()

        return MayoSample(
            g_data=g_data,
            target=target,
            max_value=max_value,
            file_name=file_name,
            min_max_value=min_max_value,
            hu_information=hu_information,
        )