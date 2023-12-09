from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import pytorch_lightning as L
import os


class CelebADataset(L.LightningDataModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.data_path = kwargs.get('data_path', os.getcwd())
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_workers = kwargs.get('num_workers', 0)
        self.pin_memory = kwargs.get('pin_memory', True)

    def prepare_data(self) -> None:
        try:
            CelebA(self.data_path, download=True)
        except RuntimeError:
            print('\033[91m'
                  'Pytorch CelebA dataset download is blocked due to daily google drive download limit.\n'
                  'Download and extract file below. Also comment out prepare_data() step: \n'
                  'https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing\n'
                  '\033[0m')

    def setup(self, stage=None) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64, antialias=True),
            transforms.CenterCrop(64),
        ])

        self.celeba_train = CelebA(self.data_path, transform=transform)

    def train_dataloader(self) -> DataLoader:
        celeba_train = DataLoader(self.celeba_train,
                                  batch_size=self.batch_size,
                                  pin_memory=self.pin_memory,
                                  num_workers=self.num_workers,
                                  shuffle=True)
        return celeba_train
