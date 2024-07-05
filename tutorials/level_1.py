"""
Derived with help from the following PyTorch Lightning Tutorials:
Basic Model: https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html
Running with SLURM: https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html

Instructions for running on Perlmutter:
> module load python
> module load pytorch
> export NUM_NODES=4 # Can use up to 4 nodes in interactive mode on Perlmutter
> export TOTAL_GPUS=$((${NUM_NODES}*4))
> export SRUN_CPUS_PER_TASK=32
> salloc --nodes $NUM_NODES --qos interactive --time 01:00:00 --constraint gpu --gpus $TOTAL_GPUS --account=m3562_g --ntasks-per-node=4 --cpus-per-task=$SRUN_CPUS_PER_TASK
> srun --ntasks-per-node=4 --gpus $TOTAL_GPUS --cpus-per-task=$SRUN_CPUS_PER_TASK --nodes $NUM_NODES python3 tutorials/level_1.py

Known Issue:
The following warning prints when running on an interactive node. This may not be a problem and may disappear when we try with a batch script.
The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python3 tutorials/level_1.py ...
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DDPStrategy

torch.set_float32_matmul_precision('medium')
# torch.set_float32_matmul_precision('high')

# Define the PyTorch nn.Modules

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64, 3))
    
    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))
    
    def forward(self, x):
        return self.l1(x)

# Define a LightningModule

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
# Define the training dataset

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, num_workers=16)

# Instantiate the model
autoencoder = LitAutoEncoder(Encoder(), Decoder()) 


cluster_environment = SLURMEnvironment()
strategy = DDPStrategy(find_unused_parameters=False,
                       cluster_environment=cluster_environment)

# Train the model
trainer = L.Trainer(accelerator="gpu", devices=4, num_nodes=int(os.environ['NUM_NODES']), strategy=strategy, max_epochs=10)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
