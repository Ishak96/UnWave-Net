import argparse
import pathlib
import h5py
import tqdm
from matplotlib import pyplot as plt

import sys
sys.path.append("../")

from src.pl_modules import CSTDataModule
from src.physics import NCCCST
from src.util import evaluate

size = 256
num_detectors = 150
emax = 100
poission_level = 1e6
gaussian_level = 0.05
add_noise = False

json_path = "../src/json/mayo.json"

reconstruction_dir = pathlib.Path("../reconstructions")
reconstruction_dir.mkdir(exist_ok=True, parents=True)
save_targets = False
save_data = False

def eval_pinv(args):
    # parse arguments
    json_path = args.json_file
    size = args.size
    num_detectors = args.num_detectors
    poission_level = args.poission_level
    add_noise = args.add_noise
    gaussian_level = args.gaussian_level
    save_targets = args.save_target
    save_data = args.save_data
    matrix_path = f"../radon/{num_detectors}"
    
    # create directories
    if save_targets:
        target_dir = pathlib.Path("../targets")
        target_dir.mkdir(exist_ok=True, parents=True)
    if save_data:
        data_dir = pathlib.Path("../data")
        data_dir.mkdir(exist_ok=True, parents=True)

    # create data module
    datamodule = CSTDataModule(json_path=json_path, size=size, num_detectors=num_detectors, emax=emax, 
                               add_noise=add_noise, poission_level=poission_level, gaussian_level=gaussian_level)
    
    cst = NCCCST(matrix_path=matrix_path, size=size, num_detectors=num_detectors, emax=emax)
    
    datamodule.setup(stage="test")

    # create metrics
    metrics = evaluate.Metrics(evaluate.METRIC_FUNCS)

    # retrive a batch of data
    dataloader = datamodule.test_dataloader()
    with open("../Pinv_evaluation.out", "w") as log_file:
        for i in tqdm.tqdm(range(len(dataloader.dataset))):
            log_file.write(f"Saving reconstructions to {reconstruction_dir}\n")
            sample = dataloader.dataset[i]
            
            target = sample.target.numpy()
            y = sample.g_data
            reconstruction = cst.adjoint_operator(y).detach().numpy()
            fname = sample.file_name
            hu_info = sample.hu_information
            min_max = sample.min_max_value
            fname = fname.split(".IMA")[0] + ".h5"

            # squeeze and unsqueeze to handle batch size of 1
            if reconstruction.ndim == 4:
                reconstruction = reconstruction.squeeze(0)
            if target.ndim == 4:
                target = target.squeeze(0)

            if reconstruction.ndim == 2:
                reconstruction = reconstruction.unsqueeze(0)
            if target.ndim == 2:
                target = target.unsqueeze(0)

            # compute metrics
            metrics.push(target, reconstruction)

            # save the reconstruction and the hu information
            rec_fname = reconstruction_dir.joinpath(fname)
            with h5py.File(rec_fname, "w") as f:
                f.create_dataset("reconstruction", data=reconstruction)
                f.create_dataset("intercept", data=hu_info.intercept)
                f.create_dataset("slope", data=hu_info.slope)
                f.create_dataset("min", data=min_max.min_value)
                f.create_dataset("max", data=min_max.max_value)

            # save target
            if save_targets:
                tar_fname = target_dir.joinpath(fname)
                with h5py.File(tar_fname, "w") as f:
                    f.create_dataset("target", data=target)

            # save data as png images
            if save_data:
                dname = f"../data/{i}.png"
                plt.imsave(dname, y.squeeze(0), cmap="hot")

        # write the metrics to the .out file
        log_file.write(f"{metrics}\n")

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", type=str, default=json_path)
    parser.add_argument("--size", type=int, default=size)
    parser.add_argument("--num-detectors", type=int, default=num_detectors)
    parser.add_argument("--emax", type=int, default=emax)
    parser.add_argument("--poission-level", type=float, default=poission_level)
    parser.add_argument("--gaussian-level", type=float, default=gaussian_level)
    parser.add_argument("--add-noise", type=bool, default=add_noise)    
    parser.add_argument("--save-target", type=bool, default=save_targets)
    parser.add_argument("--save-data", type=bool, default=save_data)

    args = parser.parse_args()

    metrics = eval_pinv(args)
    print(metrics)

if __name__ == "__main__":
    main()