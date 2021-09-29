import torch
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T

from time import time

import wandb

from src import *


N_CHANNELS = 16        # Number of CA state channels
TARGET_PADDING = 8    # Number of pixels used to pad the target image border
TARGET_SIZE = 40       # Size of the target emoji
IMAGE_SIZE = TARGET_PADDING+TARGET_SIZE
BATCH_SIZE = 4
POOL_SIZE = 512
CELL_FIRE_RATE = 0.5
N_ITERS = 50


def generator(n, device):
    return make_seed(n, N_CHANNELS-1, IMAGE_SIZE, alpha_channel=3, device=device)

pool = SamplePool(POOL_SIZE, generator)

target = read_image("images/firework.png", ImageReadMode.RGB_ALPHA).float()
target = T.Resize((TARGET_SIZE, TARGET_SIZE))(target)
target = RGBAtoFloat(target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target = target.to(device)
model = NeuralCA(N_CHANNELS, device)

model.load("Pretrained_models/firework_regenerating.pt")


optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
criterion = NCALoss(pad(target, TARGET_PADDING), torch.nn.MSELoss)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.3)

wandb.init(mode="disabled")

torch.cuda.cudnn_benchmark = False
torch.backends.cudnn.enabled = False

model.train_CA(optimizer, criterion, pool, batch_size=8, n_epochs=1, scheduler=scheduler, kind="regenerating", skip_damage=2)
_ = model.test_CA(criterion, pool.images, evolution_iters=55)

print("Benchmark type \t images/s")

start = time()
model.train_CA(optimizer, criterion, pool, batch_size=8, n_epochs=1, scheduler=scheduler, kind="regenerating", skip_damage=2)
print(f"Train w/o cudnn: {POOL_SIZE/(time()-start)}")

start = time()
_ = model.test_CA(criterion, pool.images, evolution_iters=55)
print(f"Test w/o cudnn: {POOL_SIZE/(time()-start)}")
print("\n")

torch.cuda.cudnn_benchmark = True
torch.backends.cudnn.enabled = True

model.train_CA(optimizer, criterion, pool, batch_size=8, n_epochs=1, scheduler=scheduler, kind="regenerating", skip_damage=2)
_ = model.test_CA(criterion, pool.images, evolution_iters=55)

start = time()
model.train_CA(optimizer, criterion, pool, batch_size=8, n_epochs=1, scheduler=scheduler, kind="regenerating", skip_damage=2)
print(f"Train w cudnn: {POOL_SIZE/(time()-start)}")

start = time()
_ = model.test_CA(criterion, pool.images, evolution_iters=55)
print(f"Test w cudnn: {POOL_SIZE/(time()-start)}")
print("\n")


for batch in range(3, 8):
    start = time()
    model.train_CA(optimizer, criterion, pool, batch_size=2**batch, n_epochs=1, scheduler=scheduler, kind="regenerating", skip_damage=2)
    print(f"Train w cudnn, batch size {2**batch}: {POOL_SIZE/(time()-start)}")