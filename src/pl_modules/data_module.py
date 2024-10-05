import numpy as np

from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader
from src.data import MayoData

class CSTDataModule(LightningDataModule):
    def __init__(self,
                 json_path: str,
                 size: int,
                 num_detectors: int = 100,
                 emax: int = 100,
                 add_noise: bool = True,
                 poission_level: float = 5e6,
                 gaussian_level: float = 0.05,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 distributed_sampler: bool = False,
    ):
        super().__init__()

        self.json_path = json_path
        self.size = size
        self.num_detectors = num_detectors
        self.emax = emax
        self.add_noise = add_noise
        self.poission_level = poission_level
        self.gaussian_level = gaussian_level
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = MayoData(json_path=self.json_path, mode='train', size=self.size, num_detectors=self.num_detectors, emax=self.emax,
                                          add_noise=self.add_noise, poission_level=self.poission_level, gaussian_level=self.gaussian_level
                                )
            self.val_dataset = MayoData(json_path=self.json_path, mode='val', size=self.size, num_detectors=self.num_detectors, emax=self.emax,
                                        add_noise=self.add_noise, poission_level=self.poission_level, gaussian_level=self.gaussian_level
                                )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = MayoData(json_path=self.json_path, mode='test', size=self.size, num_detectors=self.num_detectors, emax=self.emax,
                                         add_noise=self.add_noise, poission_level=self.poission_level, gaussian_level=self.gaussian_level
                                )

    def _create_data_loader(self, data_partition: str):
        if data_partition == 'train':
            dataset = self.train_dataset
        elif data_partition == 'val':
            dataset = self.val_dataset
        elif data_partition == 'test':
            dataset = self.test_dataset
        
        dataLoader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=(data_partition=='train'),
                                num_workers=self.num_workers,
        )

        return dataLoader

    def train_dataloader(self):
        return self._create_data_loader('train')

    def val_dataloader(self):
        return self._create_data_loader('val')

    def test_dataloader(self):
        return self._create_data_loader('test')

