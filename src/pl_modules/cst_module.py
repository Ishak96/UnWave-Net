import pathlib
from collections import defaultdict

import numpy as np
import h5py

import pytorch_lightning as pl
import torch
from torchmetrics.metric import Metric

from src.util import evaluate

class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.add_state("quantity", default=torch.tensor(0.0, dtype=torch.float), dist_reduce_fx="sum")
    
    def update(self, batch):
        self.quantity += batch

    def compute(self):
        return self.quantity

class CSTModule(pl.LightningModule):
    def __init__(self, save_targets=False, **kwargs):
        super().__init__()
        
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

        self.metrics = evaluate.Metrics(evaluate.METRIC_FUNCS)
        self.reconstruction_dir = pathlib.Path.cwd().joinpath("reconstructions")
        self.reconstruction_dir.mkdir(exist_ok=True, parents=True)
        
        if save_targets:
            self.target_dir = pathlib.Path.cwd().joinpath("targets")            
            self.target_dir.mkdir(exist_ok=True, parents=True)
        self.save_targets = save_targets

        self.val_outputs = []
        self.batch_index_to_log = 0

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        # check inputs
        for k in (
            "batch_idx",
            "max_value",
            "fname",
            "output",
            "target",
            "val_loss",
        ):
            if k not in outputs.keys():
                raise ValueError(f"Missing key: {k}")
            
        if outputs["output"].ndim == 4:
            outputs["output"] = outputs["output"].squeeze(0)
        if outputs["target"].ndim == 2:
            outputs["target"] = outputs["target"].unsqueeze(0)
        
        if outputs["target"].ndim == 4:
            outputs["target"] = outputs["target"].squeeze(0)
        if outputs["output"].ndim == 2:
            outputs["output"] = outputs["output"].unsqueeze(0)

        elif outputs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if outputs["target"].ndim != 3:
            raise RuntimeError("Unexpected target size from validation_step.")

        # log images to tensorboard
        if isinstance(outputs["batch_idx"], int):
            batch_indices = [outputs["batch_idx"]]
        else:
            batch_indices = outputs["batch_idx"]

        for i, batch_idx in enumerate(batch_indices):
            if batch_idx == self.batch_index_to_log:
                key = f"val_images{self.global_step}"
                output = outputs["output"][i].unsqueeze(0)
                target = outputs["target"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        max_vals = dict()

        for i, fname in enumerate(outputs["fname"]):
            output = outputs["output"][i].cpu().numpy()
            target = outputs["target"][i].cpu().numpy()
            max_val = outputs["max_value"][i].cpu().numpy()

            mse_vals[fname] = torch.tensor(
                evaluate.mse(output, target)
            )
            ssim_vals[fname] = torch.tensor(
                evaluate.ssim(output[None, ...], target[None, ...], max_val)
            ).squeeze(0)
            psnr_vals[fname] = torch.tensor(
                evaluate.psnr(output, target, max_val)
            )
            target_norms[fname] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            )
            max_vals[fname] = max_val

        # save metrics to dict
        self.val_outputs.append(
            {
                "val_loss": outputs["val_loss"],
                "mse_vals": mse_vals,
                "ssim_vals": ssim_vals,
                "psnr_vals": psnr_vals,
                "target_norms": target_norms,
                "max_vals": max_vals,
            }
        )

    def on_validation_epoch_end(self):
        # aggregate metrics
        losses = []
        mse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for log in self.val_outputs:
            losses.append(log["val_loss"].view(-1))
            
            mse_vals.update(log["mse_vals"])
            ssim_vals.update(log["ssim_vals"])
            psnr_vals.update(log["psnr_vals"])
            target_norms.update(log["target_norms"])
            max_vals.update(log["max_vals"])

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys() 
            == ssim_vals.keys()
            == psnr_vals.keys()
            == target_norms.keys()
            == max_vals.keys()
        )

        # apply means to metrics
        metrics = {"nmse": 0, "psnr": 0, "ssim": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples += 1
            mse_val = mse_vals[fname]
            target_norm = target_norms[fname]

            metrics["nmse"] += mse_val / target_norm
            metrics["psnr"] += psnr_vals[fname]
            metrics["ssim"] += ssim_vals[fname]

        # reduce metrics across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.Tensor(losses)))        

        self.log("validation_loss", val_loss / tot_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)

        # reset val_outputs
        self.val_outputs = []

        # randomly select a batch to log
        self.batch_index_to_log = np.random.randint(0, self.trainer.num_val_batches)

    def on_test_batch_end(self, outputs, batch, batch_idx):
        self.print(f"Saving reconstructions to {self.reconstruction_dir}")
    
        # save to h5py file and compute metrics
        fname = outputs["fname"][0].split(".IMA")[0] + ".h5"
        reconstruction = outputs["output"][0]
        target = outputs["target"][0]
        hu_info = outputs["hu_info"]
        min_max = outputs["min_max"]

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
        self.metrics.push(target, reconstruction)

        # save the reconstruction and the hu information
        rec_fname = self.reconstruction_dir.joinpath(fname)
        with h5py.File(rec_fname, "w") as f:
            f.create_dataset("reconstruction", data=reconstruction)
            f.create_dataset("intercept", data=hu_info.intercept.cpu())
            f.create_dataset("slope", data=hu_info.slope.cpu())
            f.create_dataset("min", data=min_max.min_value.cpu())
            f.create_dataset("max", data=min_max.max_value.cpu())

        # save target
        if self.save_targets:
            tar_fname = self.target_dir.joinpath(fname)
            with h5py.File(tar_fname, "w") as f:
                f.create_dataset("target", data=target)        

    def on_test_end(self):
        print(self.metrics)