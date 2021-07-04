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
    def __init__(self, decaying_CA: CAModel, regenerating_CA: CAModel, mutation_probability=0.6):
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
        x_new = torch.tanh(self.new_CA(x))
        return x_old * self.old_cells + x_new * self.new_cells

    def evolve(self, x, n_iters):
        with torch.no_grad():
            for i in range(n_iters):
                x = self.forward(x)
        
        return x
    
    def train_CA(self, optimizer, criterion, pool, n_epochs, scheduler=None, batch_size=4, evolution_iters=55, skip_update=2):
        self.old_CA.train()
        self.new_CA.train()

        for i in range(n_epochs):
            epoch_losses = []
            for j in range(pool.size // batch_size):
                inputs, indexes = pool.sample(batch_size)
                inputs = inputs.to(self.device)
                self.update_cell_masks(inputs)
                optimizer.zero_grad()
                
                for k in range(evolution_iters + randint(-5, 5)):
                    inputs = self.forward(inputs)
                    
                loss, idx_max_loss = criterion(inputs)
                epoch_losses.append(loss.item())
                if j%skip_update != 0:
                    idx_max_loss = None
                loss.backward()
                optimizer.step()
                pool.update(inputs, indexes, idx_max_loss)
            
            if scheduler is not None:
                scheduler.step()

            self.losses.append(np.mean(epoch_losses))
            print(f"epoch: {i+1}\navg loss: {np.mean(epoch_losses)}")
            clear_output(wait=True)


    def eval_CA(self, criterion, image_size, eval_samples=128, evolution_iters=1000, batch_size=32):
        self.evolution_losses = torch.zeros((evolution_iters), device="cpu")
        n = 0
        with torch.no_grad():
            for i in range(eval_samples // batch_size):
                inputs = make_seed(batch_size, self.new_CA.n_channels, image_size, device=self.device)
                self.update_cell_masks(inputs)
                for j in range(evolution_iters):
                    inputs = self.forward(inputs)
                    loss, _ = criterion(inputs)
                    self.evolution_losses[j] = (n*self.evolution_losses[j] + batch_size*loss.cpu()) / (n+batch_size)
                n += batch_size

        return self.evolution_losses


    def make_video(self, n_iters, video_size, fname="video.mkv", rescaling=8, init_state=None, fps=10):
        if init_state is None:
            init_state = make_seed(1, self.new_CA.n_channels, video_size)

        init_state = init_state.to(self.device)
        
        video_size *= rescaling
        video = torch.empty((n_iters, video_size, video_size, 3), device="cpu")
        rescaler = T.Resize((video_size, video_size), interpolation=T.InterpolationMode.NEAREST)
        with torch.no_grad():
            for i in range(n_iters):
                video[i] = FloattoRGB(rescaler(init_state))[0].permute(1,2,0).cpu()
                init_state = self.forward(init_state)
        
        write_video(fname, video, fps=fps)