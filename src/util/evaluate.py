from typing import Union

import numpy as np
import torch

from typing import Optional

from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class Normalization(object):
    def __init__(self,
                 min_value: float = 0.0,
                 max_value: float = 1.0,
    ):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self,
                    x: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, np.ndarray):
            min_x = np.min(x)
            max_x = np.max(x)
        else:
            min_x = torch.min(x)
            max_x = torch.max(x)            

        nom = x - min_x
        denom = max_x - min_x

        return (nom / denom) * (self.max_value - self.min_value) + self.min_value

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)

def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array((np.linalg.norm(gt - pred) ** 2) / (np.linalg.norm(gt) ** 2))

def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    return structural_similarity(gt[0], pred[0], data_range=maxval, channel_axis=False)

METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)

class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )