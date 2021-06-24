from numpy import equal
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode, write_video
import torchvision.transforms as T
import os.path
from importlib import reload

from scripts import *
from CAModel import CAModel

class DoubleCA(nn.Module):
    def __init__(self, decaying_CA: CAModel, regenerating_CA: CAModel):
        super().__init__()
        if decaying_CA.device != regenerating_CA.device:
            Exception(f"The two CAs are on different devices: " + 
                      f"decaying_CA.device: {decaying_CA.device} and " +
                      f"regenerating_CA.device: {regenerating_CA.device}")

        self.device = decaying_CA.device
        self.old_CA = decaying_CA
        self.new_CA = regenerating_CA
    
    def update_cell_masks(self, x, new_cell_probability = 0.1):
        new_cells_mask = torch.rand(x[:,:1,:,:].size(), device=self.device) < new_cell_probability
                
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