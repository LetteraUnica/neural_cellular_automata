import torch
from torchvision.io import read_image, ImageReadMode, write_video
import torchvision.transforms as T
from random import randint
from IPython.display import clear_output
import numpy as np
import pylab as pl
import wandb

from src import *

N_CHANNELS = 15        # Number of CA state channels
TARGET_PADDING = 8     # Number of pixels used to pad the target image border
TARGET_SIZE = 40       # Size of the target emoji
IMAGE_SIZE = TARGET_PADDING+TARGET_SIZE
POOL_SIZE = 512
CELL_FIRE_RATE = 0.5
PATH=''

default_config={
    'percentage':0.7,
    'lr':0.01,
    'batch_size': 80,
    'n_epochs':100,
    'kind':'persist'
    }

#improve this code to have better monitorning
wandb.init(project='mask', entity="neural_ca", config=default_config)
config=wandb.config
print(config)

torch.backends.cudnn.benchmark = True # Speeds up things
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Imports the target emoji
target = read_image(PATH+"images/firework.png", ImageReadMode.RGB_ALPHA).float()
target = T.Resize((TARGET_SIZE, TARGET_SIZE))(target)
target = RGBAtoFloat(target)
#imshow(target)
target = target.to(device)


#import the models

old_CA=NeuralCA(device=device)
new_CA=NeuralCA(device=device)

for param in old_CA.parameters():
   param.requires_grad = False

old_CA.load(PATH+'Pretrained_models/firework_growing.pt')
new_CA.load(PATH+'Pretrained_models/mask 70% persist.pt')

model = VirusCA(old_CA, new_CA, mutation_probability=config['percentage'])
model.to(device)

wandb.watch(model, log_freq=32)


#generate the pool
generator=VirusGenerator(N_CHANNELS,IMAGE_SIZE,2,model,config['percentage'])
pool = SamplePool(POOL_SIZE, generator)

# Imports the target emoji
target = read_image(PATH+"images/firework.png", ImageReadMode.RGB_ALPHA).float()
target = T.Resize((TARGET_SIZE, TARGET_SIZE))(target)
target = RGBAtoFloat(target)
target = target.to(device)

# Zero out gradients on the first CA
for param in model.old_CA.parameters():
    param.requires_grad = False

# Set up the training 
params = model.new_CA.parameters()

optimizer = torch.optim.Adam(model.new_CA.parameters(), lr=config['lr'])
criterion = NCALoss(pad(target, TARGET_PADDING))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.3)


# The actual training part
model.train_CA(
    optimizer,
    criterion,
    pool,
    batch_size=config['batch_size'],
    n_epochs=config['n_epochs'],
    scheduler=scheduler,
    skip_update=1,
    kind=config['kind'],
    n_max_losses=5,
    reset_prob=1/40)


model.new_CA.save(f"mask_"+str(config['percentage']*100)+ "%.pt",overwrite=True)
wandb.save("mask_"+str(config['percentage']*100)+ "%.pt")