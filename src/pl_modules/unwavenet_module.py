import torch

import torch.nn as nn

from src.physics import NCCCST
from src.models import UnWaveNet

from .cst_module import CSTModule

class UnWaveNetModule(CSTModule):
    def __init__(self,
                 physics_model: NCCCST,
                 num_cascades: int = 12,
                 lr: float = 0.0001,
                 weight_decay: float = 0.0,
                 **kwargs,
    ):
        super().__init__(**kwargs)

        self.physics_model = physics_model
        self.criterion = nn.MSELoss()
        self.unwavenet = UnWaveNet(num_cascades=num_cascades, **kwargs)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, y):
        return self.unwavenet(physics_model=self.physics_model, y=y)

    def training_step(self, batch, batch_idx):
        y, target = batch.g_data, batch.target

        out_image = self(y)
        loss = self.criterion(out_image, target)

        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y, target, max_value, file_name = batch.g_data, batch.target, batch.max_value, batch.file_name

        out_image = self(y)
        loss = self.criterion(out_image, target)

        return {
            "batch_idx": batch_idx,
            "max_value": max_value,
            "fname": file_name,
            "target": target,
            "output": out_image,
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        y, max_value, file_name = batch.g_data, batch.max_value, batch.file_name

        out_image = self(y)
        out_image = out_image.cpu().numpy()
        target = batch.target.cpu().numpy()

        return {
            "batch_idx": batch_idx,
            "max_value": max_value,
            "fname": file_name,
            "output": out_image,
            "target": target,
            "min_max": batch.min_max_value,
            "hu_info": batch.hu_information,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unwavenet.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(self.trainer.max_epochs * .8), gamma=.1)

        return [optimizer], [scheduler]
