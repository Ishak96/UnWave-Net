import os

from itertools import islice

from pathlib import Path

import matplotlib.pyplot as plt

import torch

from src.util import evaluate, to_hu

from pytorch_lightning import Callback, LightningModule, Trainer

class CSTDisplayCallback(Callback):
    def __init__(self,
                 every_n_epochs: int = 1,
                 path: str = "epochs/",
                 sample_index: int = 12,
                 hu_min: float = -150.,
                 hu_max: float = 250.,
                 rescale: bool = False,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs

        self.path = Path(path)
        self.path.mkdir(exist_ok=True, parents=True)
        self.sample_index = sample_index
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.rescale = rescale

    def _check_dir(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def ct_display_function(self, display_list, current_epoch):
        y, target, output, max_value, min_max_value, hu_information = display_list

        y = y.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        max_value = max_value.cpu().detach().numpy()

        psnr = evaluate.psnr(output, target, max_value)
        ssim = evaluate.ssim(output, target, max_value)
        
        target_hu = to_hu(target, min_max_value, hu_information, self.hu_min, self.hu_max, self.rescale)
        output_hu = to_hu(output, min_max_value, hu_information, self.hu_min, self.hu_max, self.rescale)

        titles = [f'Input data', 'Ground Truth', f"Reconstruction (PSNR: {psnr:.4f}, SSIM: {ssim:.4f})"]

        fig, axs = plt.subplots(1, 3, figsize=(15, 8))
        axs = axs.flatten()

        for img, ax, title in zip([y, target_hu, output_hu], axs, titles):
            ax.imshow(img[0, ...], cmap="gray")
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        if self.path is not None:
            plt.savefig(self.path / f"epoch_{current_epoch}.png")
        else:
            plt.show()

    def on_train_epoch_end(self,
                           trainer: Trainer, 
                           pl_module: LightningModule
    ):
        val_dataloader = trainer.val_dataloaders
        val_dataset = val_dataloader.dataset
        sample = next(islice(iter(val_dataset), self.sample_index, None))

        y, target, max_value = sample.g_data, sample.target, sample.max_value
        min_max_value, hu_information = sample.min_max_value, sample.hu_information
        
        y = y.to(pl_module.device).unsqueeze(0)
        target = target.to(pl_module.device).unsqueeze(0)

        with torch.no_grad():
            pl_module.eval()
            output = pl_module.forward(y)
            pl_module.train()
        
        y = y.squeeze(0)
        target = target.squeeze(0)
        output = output.squeeze(0)

        display_tup = (y, target, output, max_value, min_max_value, hu_information)
        
        self.ct_display_function(display_tup, current_epoch=trainer.current_epoch)

