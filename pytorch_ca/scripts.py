from hashlib import new
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pylab as pl
from random import randint


# Utility functions
def RGBAtoFloat(images):
    """Converts images in 0-1 range"""
    return torch.clip(images.float() / 255, 0., 1.)

def FloattoRGBA(images):
    """Converts images in 0-255 range"""
    return torch.clip((images * 255), 0, 255).type(torch.uint8)

def FloattoRGB(images):
    """Converts a 0-1 float image with an alpha channel into RGB"""
    if len(images.size()) < 4:
        images = torch.unsqueeze(images, 0)
    return torch.clip(images[:,:3,:,:] * images[:,3,:,:] * 255 + (1-images[:,3,:,:])*255, 0, 255).type(torch.uint8)

def FloattoTrain(images, apply_tanh=False):
    """Converts a 0-1 float image to [-1,1] range"""
    images = images * 2 - 1
    if apply_tanh:
        images = torch.tanh(images*2)/np.tanh(2)

    return images

def TraintoFloat(images, inverse_tanh=False):
    """Converts a [-1,1] float image to 0-1 range"""
    images = torch.clip(images, -1., 1.)
    if inverse_tanh:
        images = torch.atanh(images * np.tanh(2)) / 2
    return images * 0.5 + 0.5


def center_crop(images, size):
    """Center crops an image"""
    return T.CenterCrop(size)(images)

def pad(images, padding, fill=0.):
    return T.Pad(padding//2, fill=fill)(images)

def imshow(image, apply_center_crop=False):
    """Prints an image"""
    if apply_center_crop:
        image = center_crop(image)
    pl.imshow(np.asarray(image.cpu().permute(1,2,0)[:,:,:4]))


def make_seed(n_images, n_channels, image_size, device="cpu"):
    """Makes the seed to start the CA, i.e. a black dot"""
    start_point = torch.zeros((n_images, n_channels, image_size, image_size), device=device)
    start_point[:, 3, image_size//2, image_size//2] = 1.
    return start_point

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def side(size):
    return randint(size//3, size//2)

def make_squares(images, target_size=None, side=side):
    if target_size is None:
        target_size = images.size()[-1]
    for i in range(images.size()[0]):
        x = randint(target_size//2-target_size//4, target_size//2+target_size//4)
        y = randint(target_size//2-target_size//4, target_size//2+target_size//4)
        images[i, :, x-side(target_size)//2:x+side(target_size)//2, y-side(target_size)//2:y+side(target_size)//2] = 0.

    return images

# Custom loss function
class loss_fn:
    """Custom l2 or l1 loss function"""
    def __init__(self, target, order=2):
        self.order = order
        self.target = target
        
    def __call__(self, x):
        losses = torch.mean(torch.abs(x[:, :4, :, :] - self.target)**self.order, [1,2,3])
        idx_max_loss = torch.argmax(losses)
        loss = torch.mean(losses)
        return loss, idx_max_loss

# Sample pool dataset
class SamplePool(Dataset):
    """Samples the training images"""
    def __init__(self, pool_size, n_channels, image_size, transform = None, device="cpu"):
        self.images = make_seed(pool_size, n_channels, image_size, device)
        self.size = pool_size
        self.n_channels = n_channels
        self.image_size = image_size
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

        self.device = torch.device(device)
    
    def __len__(self):
        return self.size

    
    def __getitem__(self, idx):
        return self.transform(self.images[idx]), idx

    
    def transform(self, transformation):
        self.images = transformation(self.images)

    
    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, False)
        return self.transform(self.images[idx]), idx
    

    def update(self, new_images, idx, idx_max_loss=None):
        self.images[idx] = new_images.detach().to(self.device)
        if idx_max_loss is not None:
            self.images[idx[idx_max_loss]] = make_seed(1, self.n_channels, self.image_size)[0]