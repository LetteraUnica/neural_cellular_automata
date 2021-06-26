from numpy import equal
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode, write_video
import torchvision.transforms as T
import os.path
from importlib import reload

from scripts import *
from CA_model import *

class VirusCA(nn.Module):
    def __init__(self, decaying_CA: CAModel, regenerating_CA: CAModel, new_cell_probability=0.1):
        super().__init__()
        if decaying_CA.device != regenerating_CA.device:
            Exception(f"The two CAs are on different devices: " + 
                      f"decaying_CA.device: {decaying_CA.device} and " +
                      f"regenerating_CA.device: {regenerating_CA.device}")

        self.device = decaying_CA.device
        self.old_CA = decaying_CA
        self.new_CA = regenerating_CA
        self.new_prob = new_cell_probability
        self.losses = []
    
    def update_cell_masks(self, x, new_cell_mask=None):
        if new_cell_mask is None:
            new_cells_mask = torch.rand(x[:,:1,:,:].size(), device=self.device) < self.new_prob

        self.new_cells = new_cells_mask.float()
        self.old_cells = 1. - self.new_cells

    def forward(self, x):
        x_old = self.old_CA(x)
        x_new = self.new_CA(x)
        return x_old * self.old_cells + x_new * self.new_cells

    def evolve(self, x, n_iters):
        with torch.no_grad():
            for i in range(n_iters):
                x = self.forward(x)
        
        return x
    
    def train_CA(self, optimizer, criterion, pool, n_epochs, scheduler=None, batch_size=4, evolution_iters=45):
        self.train()
        
        self.update_cell_masks(pool.sample(batch_size))
        for i in range(n_epochs):
            epoch_losses = []
            for j in range(pool.size // batch_size):
                inputs, indexes = pool.sample(batch_size)
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                
                for k in range(evolution_iters + randint(-5, 5)):
                    inputs = self.forward(inputs)
                    
                loss, idx_max_loss = criterion(inputs)
                epoch_losses.append(loss.item())
                # if j%2 != 0:
                #     idx_max_loss = None
                loss.backward()
                optimizer.step()
                pool.update(inputs, indexes, idx_max_loss)
            
            if scheduler is not None:
                scheduler.step()

            self.losses.append(np.mean(epoch_losses))
            print(f"epoch: {i+1}\navg loss: {np.mean(epoch_losses)}")
            clear_output(wait=True)