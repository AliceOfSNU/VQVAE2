import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import gc
from tqdm import tqdm
import wandb
import numpy as np
import os

## my source
import vqvae
from dataset import FFHQDataset
import utils

USE_WANDB=False
BASE_DIR = "VQVAE"
DATA_DIR = os.path.join(BASE_DIR, 'data/ffhq_images')
MODEL_DIR = os.path.join(BASE_DIR, "model")

config = {
    "n_epochs" :5,
    "lr" :1e-3,
    "hidden_dim": 128,
    "embed_dim": 64,
    "n_embed": 128,
    "n_resblocks": 2,
    "seed": 12,
    "batch_size": 32,
    "latent_loss_weight":0.2
}
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

torch.cuda.empty_cache()
gc.collect()

# Dataset
transform = transforms.Compose(
    [
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
train_data = FFHQDataset(data_dir=DATA_DIR)
train_loader = DataLoader(train_data, batch_size=config["batch_size"], num_workers=1)

# model
model = vqvae.VQVAE(
    3,  #in channels
    config["hidden_dim"], #hidden dim
    config["embed_dim"], #embed dim
    config["n_embed"], #vocab size(dictionary embedding n)
    config["n_resblocks"] #resblocks inside encoder/decoder
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
        for i, img in enumerate(train_loader):
            out, latent_loss = model(img.to(device))
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
        
train(model, train_loader, config)