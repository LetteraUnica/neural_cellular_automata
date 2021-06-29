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

class VirusCA:
    def __init__(self, decaying_CA: CAModel, regenerating_CA: CAModel, mutation_probability=0.1):
        if decaying_CA.device != regenerating_CA.device:
            Exception(f"The two CAs are on different devices: " + 
                      f"decaying_CA.device: {decaying_CA.device} and " +
                      f"regenerating_CA.device: {regenerating_CA.device}")

        self.device = decaying_CA.device
        self.old_CA = decaying_CA
        self.new_CA = regenerating_CA
        self.new_prob = mutation_probability
        self.losses = []
    
    def update_cell_masks(self, x, mutation_mask=None):
        if mutation_mask is None:
            mutation_mask = torch.rand(x[:,:1,:,:].size(), device=self.device) < self.new_prob

        self.new_cells = mutation_mask.to(self.device).float()
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
    
    def train_CA(self, optimizer, criterion, pool, n_epochs, scheduler=None, batch_size=4, evolution_iters=55):
        self.old_CA.train()
        self.new_CA.train()
        # self.update_cell_masks(pool.sample(batch_size)[0])

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
                if j%2 != 0:
                    idx_max_loss = None
                loss.backward()
                optimizer.step()
                pool.update(inputs, indexes, idx_max_loss)
            
            if scheduler is not None:
                scheduler.step()

            self.losses.append(np.mean(epoch_losses))
            print(f"epoch: {i+1}\navg loss: {np.mean(epoch_losses)}")
            clear_output(wait=True)




class VirusCA2:
    def __init__(self, decaying_CA: CAModel, regenerating_CA: CAModel, mask_CA: CAModel):
        if decaying_CA.device != regenerating_CA.device:
            Exception(f"The two CAs are on different devices: " + 
                      f"decaying_CA.device: {decaying_CA.device} and " +
                      f"regenerating_CA.device: {regenerating_CA.device}")

        self.device = decaying_CA.device
        self.old_CA = decaying_CA
        self.new_CA = regenerating_CA
        self.mask_CA = mask_CA
        self.losses = []

    def forward(self, x):
        x_old = self.old_CA(x)
        x_new = torch.tanh(self.new_CA(x))
        self.new_cells = torch.sigmoid(self.mask_CA(x)[:, -1:, :, :])

        return x_old * (1. - self.new_cells) + x_new * self.new_cells

    def evolve(self, x, n_iters):
        with torch.no_grad():
            for i in range(n_iters):
                x = self.forward(x)
        
        return x


    def train_CA(self, optimizer, criterion, pool, n_epochs, scheduler=None, batch_size=4, evolution_iters=55):
        self.old_CA.train()
        self.new_CA.train()
        self.mask_CA.train()

        for i in range(n_epochs):
            epoch_losses = []
            for j in range(pool.size // batch_size):
                inputs, indexes = pool.sample(batch_size)
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                
                for k in range(evolution_iters + randint(-5, 5)):
                    inputs = self.forward(inputs)
                    
                loss, idx_max_loss = criterion(inputs, self.new_cells)
                epoch_losses.append(loss.item())
                if j%2 != 0:
                    idx_max_loss = None
                loss.backward()
                optimizer.step()
                pool.update(inputs, indexes, idx_max_loss)
            
            if scheduler is not None:
                scheduler.step()

            self.losses.append(np.mean(epoch_losses))
            print(f"epoch: {i+1}\navg loss: {np.mean(epoch_losses)}")
            clear_output(wait=True)



class loss_fn2:
    """Custom l2 or l1 loss function"""
    def __init__(self, target, order=2, l=1e-2):
        self.order = order
        self.target = target
        self.l = l
        
    def __call__(self, x, mask):
        losses = torch.mean(torch.abs(x[:, :4, :, :] - self.target)**self.order, [1,2,3])
        idx_max_loss = torch.argmax(losses)
        loss = torch.mean(losses) + self.l*torch.mean(mask**2)
        return loss, idx_max_loss