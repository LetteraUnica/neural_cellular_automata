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
    'percentage':0.97,
    'gamma':0.9404,
    'lr1':0.002,
    'lr2':0.002,
    'batch_size': 85,
    'n_epochs':300,
    'n_max_loss_ratio':8,
    'step_size':48.328
    }

#improve this code to have better monitorning
wandb.init(project='NeuralCA', entity="neural_ca", config=default_config)
config=wandb.config
print(config)

torch.backends.cudnn.benchmark = True # Speeds up things
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#import the models
model = MultipleCA(N_CHANNELS, n_CAs=2, device=device)

model.CAs[0].load_state_dict(torch.load(PATH+'Pretrained_models/firework_growing.pt', map_location=device))
model.CAs[1].load_state_dict(torch.load(PATH+'Pretrained_models/glamorous-sweep-21.pt', map_location=device))

model.to(device)

wandb.watch(model, log_freq = 32)


#generate the pool
generator=VirusGenerator(N_CHANNELS,IMAGE_SIZE,2,model,config['percentage'])
pool = SamplePool(POOL_SIZE, generator)

# Imports the target emoji
target = read_image(PATH+"images/firework.png", ImageReadMode.RGB_ALPHA).float()
target = T.Resize((TARGET_SIZE, TARGET_SIZE))(target)
target = RGBAtoFloat(target)
target = target.to(device)

# Zero out gradients on the first CA
#for param in model.CAs[0].parameters():
#    param.requires_grad = False

# Set up the training 
params = model.CAs[1].parameters()
optimizer = torch.optim.Adam(params, lr=config['lr1'])
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,config['lr1'],config['lr1']+config['lr2'],config['step_size'],gamma=config['gamma'], cycle_momentum=False)
criterion = NCALoss(pad(target, TARGET_PADDING), torch.nn.MSELoss, alpha_channels=[15, 16])


# The actual training part
model.train_CA(
    optimizer,
    criterion,
    pool,
    batch_size=config['batch_size'],
    n_epochs=config['n_epochs'],
    scheduler=scheduler,
    skip_update=1,
    kind="regenerating",
    n_max_losses=config['batch_size'] // config['n_max_loss_ratio'],
    skip_damage=2,
    reset_prob=1/40)


model.CAs[1].save(f"model.pt",overwrite=True)
wandb.save('model.pt')


converter=[tensor_to_RGB(function="RGBA",CA=model),
           tensor_to_RGB(function=-2,CA=model),
           tensor_to_RGB(function=-1,CA=model)]

seed=make_seed(1,N_CHANNELS,IMAGE_SIZE,n_CAs=2,alpha_channel=-2,device=device)
fname=["RGBA.mp4","-2.mp4","-1.mp4"]

video,init_state=make_video(model,30,seed,converter=converter)
init_state=add_virus(init_state,-2,-1,0.995)
make_video(model,50,init_state,fname=fname,initial_video=video,converter=converter)

for name in fname:
    wandb.log({name : wandb.Video(name,fps=10,format='mp4')})