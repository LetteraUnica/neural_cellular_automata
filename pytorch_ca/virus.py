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

config={
    'percentage':0.97,
    'lr':2e-3,
    'batch_size': 256,
    'n_epochs':60
    }



torch.backends.cudnn.benchmark = True # Speeds up things
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login() #does the login, i dunno if it is really needed



# Imports the target emoji
target = read_image("images/firework.png", ImageReadMode.RGB_ALPHA).float()
target = T.Resize((TARGET_SIZE, TARGET_SIZE))(target)
target = RGBAtoFloat(target)
#imshow(target)
target = target.to(device)

#improve this code to have better monitorning
wandb.init(project='virus',entity="boh",config=config)



#import the models
model = MultipleCA(N_CHANNELS, n_CAs=2, device=device)

model.CAs[0].load_state_dict(torch.load('Pretrained_models/firework_growing.pt', map_location=device))
model.CAs[1].load_state_dict(torch.load('Pretrained_models/switch.pt', map_location=device))

model.to(device)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)

wandb.watch(model)


#generate the pool
generator=VirusGenerator(N_CHANNELS,IMAGE_SIZE,2,model,config['percentage'])
pool = SamplePool(POOL_SIZE, generator)

# Imports the target emoji
target = read_image("images/firework.png", ImageReadMode.RGB_ALPHA).float()
target = T.Resize((TARGET_SIZE, TARGET_SIZE))(target)
target = RGBAtoFloat(target)
target = target.to(device)

#set up the training 
params=model.CAs[1].parameters()
optimizer = torch.optim.Adam(params, lr=config['lr'])
criterion = NCALoss(pad(target, TARGET_PADDING), torch.nn.MSELoss, alpha_channels=[15, 16])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45], gamma=0.3)


#The actual training part
model.train_CA(
    optimizer,
    criterion,
    pool,
    batch_size=config['batch_size'],
    n_epochs=config['n_epochs'],
    scheduler=scheduler,
    skip_update=1,
    kind="regenerating",
    n_max_losses=config['batch_size'] // 4,
    skip_damage=2)

