from dataset import LmdbDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import gc
from tqdm import tqdm
import wandb
import numpy as np
import os
import json

from pixelsnail import PixelSnail
from vqvae import SingleVQVAE, VQVAE
import utils


USE_WANDB=True

config = {
    "n_epochs" :50,
    "lr" :1e-3,
    "hidden_dim": 128,
    "n_layers": 2,
    "n_resblocks": 4,
    "n_output_layers":2,
    "batch_size": 32,
    "attn_embed_dim":128,
    "attn_n_heads":4,
    "down_kernel": (2, 5),
    "downright_kernel":(2, 3),
    "hidden_kernel":(3, 5),
    "run_id": "single/PRIOR_single",
    "vae_id": "single/VAE_single",
    "seed": 12,
    "note": "prior for default SingleVAE"
}


BASE_DIR = "VQVAE"
SNAIL_MODEL_DIR = os.path.join(BASE_DIR, "model/snail_prior")
VAE_MODEL_DIR = os.path.join(BASE_DIR, "model", config["vae_id"])
DATA_DIR = os.path.join(VAE_MODEL_DIR, "code.lmdb")
run_id = config["run_id"]
save_dir = os.path.join(SNAIL_MODEL_DIR, run_id)

if USE_WANDB:
    run = wandb.init(
        name = run_id, 
        reinit = True, 
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "PixelCNN_VQVAE", 
        config=config
    )
        
# generates priors with pixelsnail
def generate_samples(model, N, size, device, cond = None):
    # return 1, H, W generated samples from model.
    model.eval()
    result = torch.zeros(N, *size, dtype=torch.int64).to(device)
    for i in range(size[0]):
        for j in range(size[1]):
            out = model(result, label_cond=cond)
            prob = out[:, :, i, j].softmax(dim = 1)
            result[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)
            
    model.train()
    return result

def train(model, VAE_model, train_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(reduction='mean')
    model.train()
    VAE_model.eval() # we are not training this
    
    n_epochs = config["n_epochs"]
    best_accuracy = 0.0
    for epoch in range(n_epochs):
        accuracy, train_loss = 0.0, 0.0
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
        for i, (img, filename) in enumerate(train_loader):
            img = img.to(device) #B, H, W LongTensor
            out = model(img)
            
            loss = criterion(out, img)
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = out.max(1)
            correct = (pred == img).float()
            accuracy += correct.sum().item() / img.numel()

            # log
            batch_bar.set_postfix(
                loss="{:.04f}".format(train_loss / (i + 1)),
                accuracy="{:.04f}".format(accuracy / (i + 1))
                #lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])
            )
            batch_bar.update()
            
            # debug only
            #if i == 10: break
        
        train_loss /= len(train_loader)
        accuracy /= len(train_loader)
        batch_bar.close() 
        print(f"epoch{epoch}/{n_epochs} acc:{accuracy:.04f}\tloss:{train_loss:.04f}")
        
        samples = generate_samples(model, 10, (16, 16), device) #LongTensor
        samples = VAE_model.generate(samples.detach())
        samples = samples.cpu().numpy()
        samples = np.transpose(samples, (0, 2, 3, 1)) #channel last
        
        if USE_WANDB:
            wandb.log({
                'train_loss': train_loss,
                'samples': [wandb.Image(sample) for sample in samples]
            })
        
        # checkpointing
        if epoch==0:
            # saving and loading
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            utils.save_model(save_dir, epoch, model, optimizer, stats_dict={
                "accuracy": accuracy,
                "config": config
            })
    if USE_WANDB:
        run.finish()
        
if __name__ == "__main__":
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    torch.cuda.empty_cache()
    gc.collect()

    dataset = LmdbDataset(DATA_DIR)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1)

    device = torch.device("cuda")
    print("training on device: ", device)
    
    # pixelsnail for mnist digits = (28, 28), 256 channels for each pixel
    Snailmodel = PixelSnail(256, hidden_dim=config["hidden_dim"], target_size=(16, 16),
                    n_layers = config["n_layers"], n_resblocks=config["n_resblocks"], n_output_layers=config["n_output_layers"], 
                    down_kernel = config["down_kernel"], 
                    downright_kernel=config["downright_kernel"], 
                    hidden_kernel=config["hidden_kernel"])
    Snailmodel = Snailmodel.to(device)

    # VAEmodel
    with open(os.path.join(VAE_MODEL_DIR, "config.json"), "r") as f:
        VAE_config = json.load(f)
        
    VAE_model = SingleVQVAE(
            3,  #in channels
            VAE_config["hidden_dim"], #hidden dim
            VAE_config["embed_dim"], #embed dim
            VAE_config["n_embed"], #vocab size(dictionary embedding n)
            VAE_config["n_resblocks"] #resblocks inside encoder/decoder
        )
    VAE_model = VAE_model.to(device)

    # load weights
    model, _, specs = utils.load_model(os.path.join(VAE_MODEL_DIR, "epoch_best.pth"), VAE_model)
    
    train(Snailmodel, VAE_model, train_loader, config)