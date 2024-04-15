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
import argparse
import json

## my source
import vqvae
from dataset import FFHQDataset, CatsDataset
import utils

USE_WANDB=True
BASE_DIR = "VQVAE"
FFHQ_DATA_DIR = os.path.join(BASE_DIR, 'data/ffhq_images')
FFHQ_LABELS_DIR = os.path.join(BASE_DIR, 'data/ffhq-features-dataset-master')
CATS_DATA_DIR = os.path.join(BASE_DIR, 'data/cat_faces/cats')
MODEL_DIR = os.path.join(BASE_DIR, "model/default")

config={
    "n_epochs": 20,
    "lr": 0.0003,
    "hidden_dim": 128,
    "embed_dim": 16,
    "n_embed": 256,
    "n_resblocks": 2,
    "seed": 12,
    "batch_size": 32,
    "latent_loss_weight": 0.25,
    "run_id": "VAE2_256x16",
    "note": "codebook size 128 of 16 dims",
    "model": "default",
    "conditions": [],
    "aux_tasks":[],
    "aux_loss_weights":{}
}

run_id = config["run_id"]
save_dir = os.path.join(MODEL_DIR, run_id)

if USE_WANDB:
    run = wandb.init(
        name = config["run_id"], 
        reinit = True, 
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "PixelCNN_VQVAE", 
        config=config
    )

def train(model, train_loader, config):
    # posterior log likelihood log p(x|z)
    criterion = nn.MSELoss()
    latent_loss_weight = config["latent_loss_weight"]
    # define optimzier
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # define scheduler
    
    n_epochs = config["n_epochs"]
    best_loss = 1e6
    for epoch in range(n_epochs):
        avg_loss, avg_commitment_loss, avg_reconstr_loss = 0.0, 0.0, 0.0
        avg_aux_losses = {k:0.0 for k in config["aux_tasks"]}
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
        
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device) #C, H, W
            out, commitment_loss, aux_losses, t_idxs, b_idxs= model(img, label)
            reconstr_loss = criterion(out, img)
            #commitment_loss = commitment_loss.mean()
            # codebook loss is calculated inside quantizer.
            # vae loss terms have two components:
            loss = reconstr_loss + latent_loss_weight * commitment_loss
            # auxiliary losses
            for key in config["aux_tasks"]:
                loss = loss + config["aux_loss_weights"][key]*aux_losses[key]
                avg_aux_losses[key] += aux_losses[key].item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # log
            avg_loss += loss.item()
            avg_reconstr_loss += reconstr_loss.item()
            avg_commitment_loss += commitment_loss.item()
            batch_bar.set_postfix(
                loss="{:.05f}".format(avg_loss / (i + 1)),
                commitment_loss="{:.05f}".format(avg_commitment_loss / (i + 1)),
                reconstr_loss="{:.05f}".format(avg_reconstr_loss / (i + 1))
                #lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])
            )
            batch_bar.update()
            
            # cleanup
            del img, out
            torch.cuda.empty_cache()
            
            #debugonly
            #if i == 10: break
        # train summary
        avg_loss /= len(train_loader)
        avg_commitment_loss /= len(train_loader)
        avg_reconstr_loss /= len(train_loader)
        avg_aux_losses = {k:v/len(train_loader) for k, v in avg_aux_losses}
        # log each epoch
        batch_bar.close()
        print(f"epoch{epoch}/{n_epochs} loss:{avg_loss:.04f}\treconstruction_loss:{avg_reconstr_loss:.04f}\tcommitement_loss:{avg_commitment_loss:.04f}")
        
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            out, commitment_loss, aux_losses, t_idxs, b_idxs = model(img, label)
            break
        reconstr = np.clip(out[:10].permute(0, 2, 3, 1).detach().cpu().numpy(), 0.0, 1.0)
        plotter = lambda ax, data: ax.pcolor(data)
        t_idxs = t_idxs[:10].detach().cpu().numpy()
        t_idxs = utils.plt_to_numpy(t_idxs, plotter)
        b_idxs = b_idxs[:10].detach().cpu().numpy()
        b_idxs = utils.plt_to_numpy(b_idxs, plotter)
        gt = img[:10].permute(0, 2, 3, 1).detach().cpu().numpy()
        
        if USE_WANDB:
            wandb.log({
                'train_loss': avg_loss,
                'train_commitment_loss': avg_commitment_loss,
                'train_reconstr_loss': avg_reconstr_loss,
                'reconstr_img': [wandb.Image(ex) for ex in reconstr],
                'gt_img': [wandb.Image(ex) for ex in gt],
                'top_codes': [wandb.Image(ex) for ex in t_idxs],
                'bottom_codes': [wandb.Image(ex) for ex in b_idxs],
                **avg_aux_losses,
            })
        
        # initialize save dirs
        if epoch==0:
            # saving and loading
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
        if avg_loss < best_loss:
            best_loss = avg_loss
            print("best loss!")
            utils.save_model(save_dir, "best", model, optimizer, stats_dict={
                "train_loss": avg_loss,
                "config": config
            })
        if epoch > 0 and epoch % 5== 0:
            utils.save_model(save_dir, epoch, model, optimizer, stats_dict={
                "train_loss": avg_loss,
                "config": config
            })
            
    if USE_WANDB:
        run.finish() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument('--single', help='run the smaller version', default=False, action='store_true')
    
    args = parser.parse_args()
    if args.single:
        config["model"] = "single"
    else:
        config["model"] = "default"
        
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    torch.cuda.empty_cache()
    gc.collect()


    # model
    if config["model"] == "default":
        train_data = FFHQDataset(data_dir=FFHQ_DATA_DIR, labels_dir=FFHQ_LABELS_DIR)
        train_loader = DataLoader(train_data, batch_size=config["batch_size"], num_workers=0)
        model = vqvae.VQVAE(
            3,  #in channels
            config["hidden_dim"], #hidden dim
            config["embed_dim"], #embed dim
            config["n_embed"], #vocab size(dictionary embedding n)
            config["n_resblocks"], #resblocks inside encoder/decoder,
            conditioned=(len(config["conditions"]) > 0),
            aux_tasks=(len(config["aux_tasks"]) > 0),
        )
    elif config["model"] == "single":
        train_data = CatsDataset(data_dir=CATS_DATA_DIR)
        train_loader = DataLoader(train_data, batch_size=config["batch_size"], num_workers=0)
        model = vqvae.SingleVQVAE(
            3,  #in channels
            config["hidden_dim"], #hidden dim
            config["embed_dim"], #embed dim
            config["n_embed"], #vocab size(dictionary embedding n)
            config["n_resblocks"] #resblocks inside encoder/decoder
        )

    device = torch.device("cuda")
    model = model.to(device)
    print("training on device: ", device)
    train(model, train_loader, config)
    