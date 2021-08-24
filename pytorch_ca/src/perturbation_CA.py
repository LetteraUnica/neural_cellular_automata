import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode, write_video
import torchvision.transforms as T
import os.path
from importlib import reload

from scripts import *
from CA_model import *


class PerturbationCA:
    def __init__(self, decaying_CA: CAModel, regenerating_CA: CAModel, mask_CA=None):
        if decaying_CA.device != regenerating_CA.device:
            Exception(f"The two CAs are on different devices: " + 
                      f"decaying_CA.device: {decaying_CA.device} and " +
                      f"regenerating_CA.device: {regenerating_CA.device}")

        self.device = decaying_CA.device
        self.old_CA = decaying_CA
        self.new_CA = regenerating_CA
        self.mask_CA = mask_CA
        self.new_cells = None
        self.losses = []


    def forward(self, x):
        pre_life_mask = self.new_CA.get_living_mask(x)

        dx_new = self.new_CA.compute_dx(x)
        dx_old = self.old_CA.compute_dx(x)

        x += dx_new + dx_old
        post_life_mask = self.new_CA.get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask

        self.new_cells = dx_new * life_mask.float()

        return x * life_mask.float()

    def evolve(self, x, n_iters):
        with torch.no_grad():
            for i in range(n_iters):
                x = self.forward(x)

        return x


    def train_CA(self, optimizer, criterion, pool, n_epochs, scheduler=None,
                 batch_size=4, evolution_iters=55, kind="persistent", square_side=20):
        self.old_CA.train()
        self.new_CA.train()

        for i in range(n_epochs):
            epoch_losses = []
            for j in range(pool.size // batch_size):
                inputs, indexes = pool.sample(batch_size)
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                
                for k in range(evolution_iters + randint(-5, 5)):
                    inputs = self.forward(inputs)
                    criterion.add_mask(self.new_cells)
                    
                loss, idx_max_loss = criterion(inputs)
                epoch_losses.append(loss.item())
                if j%4 != 0:
                    idx_max_loss = None
                loss.backward()
                optimizer.step()
                if kind == "regenerating":
                    inputs = inputs.detach()
                    inputs = make_squares(inputs, side=square_side)
                if kind != "growing":
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
                for j in range(evolution_iters):
                    inputs = self.forward(inputs)
                    loss, _ = criterion(inputs)
                    self.evolution_losses[j] = (n*self.evolution_losses[j] + batch_size*loss.cpu()) / (n+batch_size)
                n += batch_size

        return self.evolution_losses


    def make_video(self, n_iters, video_size, fname="video.mkv", rescaling=8, init_state=None, fps=10, make_square=False):
        if init_state is None:
            init_state = make_seed(1, self.new_CA.n_channels, video_size)

        init_state = init_state.to(self.device)
        
        video_size *= rescaling
        video = torch.empty((n_iters, video_size, video_size, 3), device="cpu")
        video_mask = torch.empty((n_iters, video_size, video_size, 3), device="cpu")
        rescaler = T.Resize((video_size, video_size), interpolation=T.InterpolationMode.NEAREST)
        with torch.no_grad():
            for i in range(n_iters):
                if make_square and i == 60:
                    init_state = make_squares(init_state, side=20)

                video[i] = FloattoRGB(rescaler(init_state))[0].permute(1,2,0).cpu()

                if self.new_cells is None:
                    self.new_cells = torch.zeros((1, self.new_CA.n_channels, video_size, video_size), device=self.device)

                frame = torch.mean(rescaler(self.new_cells**2), dim=[0,1]).cpu()
                video_mask[i] = FloattoRGB(FloattoGrayscale(frame))[0].permute(1,2,0)
                # video_mask[i] = frame

                init_state = self.forward(init_state)
        
        write_video(fname, video, fps=fps)
        write_video(fname.split(".")[0]+"_mask.mkv", video_mask, fps=fps)



class loss_fn2:
    """Custom l2 or l1 loss function"""
    def __init__(self, target, order=2, l=1e-2):
        self.order = order
        self.target = target
        self.l = l
        self.mask = torch.tensor([0.]).cuda()
        self.N = torch.tensor([0]).cuda()

    def __call__(self, x, mask=None):
        if mask is not None:
            self.add_mask(mask)
        losses = torch.mean(torch.abs(x[:, :4, :, :] - self.target)**self.order, [1,2,3])
        idx_max_loss = torch.argmax(losses)
        loss = torch.mean(losses) + self.l*self.mask/self.N
        self.mask = 0.
        self.N = 0
        return loss, idx_max_loss

    def add_mask(self, mask):
        self.mask += torch.mean(mask**2)
        self.N += 1