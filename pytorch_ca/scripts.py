import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pylab as pl


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

def pad(images, padding):
    return T.Pad(padding//2)(images)

def imshow(image, apply_center_crop=False):
    """Prints an image"""
    if apply_center_crop:
        image = center_crop(image)
    pl.imshow(np.asarray(image.cpu().permute(1,2,0)[:,:,:4]))


def make_seed(n_images, n_channels, image_size):
    """Makes the seed to start the CA, i.e. a black dot"""
    start_point = torch.zeros((n_images, n_channels, image_size, image_size))
    start_point[:, 3, image_size//2, image_size//2] = 1.
    return start_point


# Sample pool dataset
class SamplePool(Dataset):
    """Samples the training images"""
    def __init__(self, pool_size, n_channels, image_size):
        self.images = make_seed(pool_size, n_channels, image_size)
        self.size = pool_size
        self.n_channels = n_channels
        self.image_size = image_size
    
    
    def __len__(self):
        return self.size

    
    def __getitem__(self, idx):
        return self.images[idx], idx

    
    def transform(self, transformation):
        self.images = transformation(self.images)

    
    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, False)
        return self.images[idx], idx
    

    def update(self, new_images, idx):
        self.images[idx] = new_images