from typing import Union, NamedTuple

import argparse
from pathlib import Path

import h5py
import pydicom
from PIL import Image

import numpy as np
import torch

try:
    from src.util import Normalization, ssim, psnr, nmse
except:
    from evaluate import Normalization, ssim, psnr, nmse

class HuInformation(NamedTuple):
    intercept: float
    slope: float

class MinMaxValue(NamedTuple):
    min_value: float
    max_value: float

def get_hu_information(data: pydicom.dataset.FileDataset):
    intercept = data[('0028', '1052')].value
    slope = data[('0028', '1053')].value

    return HuInformation(intercept, slope)

def to_hu(img, min_max_value, hu_information, hu_min=-150., hu_max=250., rescale=False):
    img_norm = Normalization(min_value=min_max_value.min_value, max_value=min_max_value.max_value)(img)
    img_hu = window_image(img_norm, hu_information.intercept, hu_information.slope, hu_min, hu_max, rescale=rescale)
    
    return img_hu

def window_image(img: Union[np.ndarray, torch.Tensor],
                 intercept: float,
                 slope: float,
                 min_hu: float,
                 max_hu: float,
                 rescale: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    
    img = (img*slope + intercept) #for translation adjustments given in the dicom file. 

    img[img<min_hu] = min_hu #set img_min for all HU levels less than minimum HU level
    img[img>max_hu] = max_hu #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - min_hu) / (max_hu - min_hu)*255 #rescale to 0-255 range
    
    return img

def hu_visualization(args):
    reconstructions_dir = args.reconstructions_dir
    target_dir = args.target_dir

    visual_dir = args.visual_dir
    visual_dir.mkdir(exist_ok=True, parents=True)

    # Get target files
    target_files = target_dir.glob("*.h5")
    target_images = {}
    for target_file in target_files:
        with h5py.File(target_file, "r") as f:
            target_images[target_file.stem] = np.squeeze(f["target"][()])
    print(f"Found {len(target_images)} target images.")
    for index, recon_file in enumerate(reconstructions_dir.glob("*.h5")):
        # Read h5py file
        with h5py.File(recon_file, "r") as f:
            recon = np.squeeze(f["reconstruction"][()])
            hu_info = HuInformation(f["intercept"][()], f["slope"][()])
            min_max_value = MinMaxValue(f["min"][()], f["max"][()])

        # Get corresponding target iamge
        target = target_images[recon_file.stem] * min_max_value.max_value

        # Target and recon should have the same shape of (1, n, n)
        target = np.expand_dims(target, axis=0)
        recon = np.expand_dims(recon, axis=0)

        # Compute SSIM, PSNR and NMSE
        ssim_val = ssim(target, recon)
        psnr_val = psnr(target, recon)
        nmse_val = nmse(target, recon)

        # Format filename as "index_pred_{nmse_val}_{ssim_val}_{psnr_val}.png"
        filename = f"{index}_pred_{nmse_val}_{ssim_val}_{psnr_val}.png"
        fname = visual_dir.joinpath(filename)

        # Format to HU
        target_hu = to_hu(target, min_max_value, hu_info, hu_min=args.hu_min, hu_max=args.hu_max, rescale=min_max_value.max_value!=255.0)
        output_hu = to_hu(recon, min_max_value, hu_info, hu_min=args.hu_min, hu_max=args.hu_max, rescale=min_max_value.max_value!=255.0)

        # Save image
        if args.save_target:
            target_fname = fname.parent.joinpath(f"{index}_target.png")
            target_hu = Image.fromarray(target_hu.squeeze().astype("uint8"))
            target_hu.save(target_fname)

        output_hu = Image.fromarray(output_hu.squeeze().astype("uint8"))
        output_hu.save(fname)

        print(f"Saved {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reconstructions_dir",
        type=Path,
        default="../../reconstructions",
        help="Path to the directory containing the reconstructions.",
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        default="../../targets/",
        help="Path to the directory containing the targets",
    )
    parser.add_argument(
        "--visual_dir",
        type=Path,
        default="../../visual",
        help="Path to the directory containing the visualizations.",
    )
    parser.add_argument(
        "--hu_min",
        type=float,
        default=-150.,
        help="Minimum HU value.",
    )
    parser.add_argument(
        "--hu_max",
        type=float,
        default=250.,
        help="Maximum HU value.",
    )    
    parser.add_argument(
        "--save_target",
        default=False,
        help="Whether to save the target image.",
    )
    args = parser.parse_args()

    hu_visualization(args)
