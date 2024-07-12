import lightning as L
import torch
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str="./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    
    def prepare_data(self):
        # only run on the main process
        # use for downloading, tokenizing
        # do not assign states here, e.g. self.x = y because won't be available to other processes

        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage: str):
        # Data operations to perform on every GPU, set state here
        # Do things like count number of classes, build vocabulary, train/val/test splits,
        # create datasets, apply transforms

        # Method expects a stage argument

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )
        
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32, num_workers=16)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32, num_workers=16)
    