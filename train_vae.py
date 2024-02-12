import vqvae
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import utils
import gc
from tqdm import tqdm
import wandb
import numpy as np
import os

USE_WANDB=True
BASE_DIR = "VQVAE"
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, "model")

config = {
    "n_epochs" :5,
    "lr" :1e-3,
    "hidden_dim": 128,
    "n_resblocks": 2,
}
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

torch.cuda.empty_cache()
gc.collect()

# Dataset
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
train_data = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, num_workers=1)

# model
model = vqvae.VQVAE(
    3,
    config["hidden_dim"],
    config["n_resbloks"]
)
device = torch.device("cuda")
model = model.to(device)
print("training on device: ", device)

def train(model, train_loader, config):
    # define loss
    criterion = nn.MSELoss()
    latent_loss_weight = config["latent_loss_weight"]
    # define optimzier
    optimizer = optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # define scheduler
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    
    n_epochs = config["n_epochs"]
    for epoch in range(n_epochs):
        avg_loss = 0.0
        for i, (img, label) in enumerate(train_loader):
            out, latent_loss = model(img)
            reconstr_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            loss = reconstr_loss + latent_loss_weight * latent_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # log
            avg_loss += loss.item()
            batch_bar.set_postfix(
                loss="{:.04f}".format(loss / (i + 1)),
                reconstr_loss="{:.04f}".format(reconstr_loss / (i + 1))
                #lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])
            )
            batch_bar.update()
        
        # train summary
        avg_loss /= len(train_loader)
        
        # log each epoch
        batch_bar.close()
        print(f"epoch{epoch}/{n_epochs} loss:{avg_loss:.04f}")
        
train(model, train_loader, config["n_epochs"])