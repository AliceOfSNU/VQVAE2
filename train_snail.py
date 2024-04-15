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
## Generating samples while training will take up more memory
## especially because this instantiates the encoder model as well
## run eval on trained model, instead
GENERATE_SAMPLES=False

config = {
    "n_epochs" :15,
    "lr" :2e-4, #default mode
    "hidden_dim":256,
    "n_layers": 3,
    "n_resblocks": 3,
    "n_output_layers":2,
    "n_cond_embed": 0,
    "n_cond_img": 256,
    "batch_size": 16,
    "attn_embed_dim":256,
    "attn_n_heads":8,
    "n_embed": 256,
    "input_size": (64, 64), #input image size
    "down_kernel": (2, 5),
    "downright_kernel":(2, 3),
    "hidden_kernel":(3, 5),
    "run_id": "default/VAE2_256x16_bottom_2",
    "vae_id": "default/VAE2_256x16",
    "seed": 12,
    "note": "prior(bottom) for VAE2 with smaller codebook",
    "top_n_bottom": True,
    "conditions":[]
}


BASE_DIR = "VQVAE"
SNAIL_MODEL_DIR = os.path.join(BASE_DIR, "model/snail_prior", config["run_id"])
VAE_MODEL_DIR = os.path.join(BASE_DIR, "model", config["vae_id"])
DATA_DIR = os.path.join(VAE_MODEL_DIR, "code.lmdb")
LABELS_DIR = os.path.join(BASE_DIR,"data/ffhq-features-dataset-master")
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
def generate_samples(model, N, size, device, img_cond = None, label_cond = None):
    # return 1, H, W generated samples from model.
    batch_bar = tqdm(total=size[0]*size[1], dynamic_ncols=True, leave=False, position=0, desc='Generate z')
    with torch.inference_mode():
        result = torch.zeros(N, *size, dtype=torch.int64).to(device)
        for i in range(size[0]):
            for j in range(size[1]):
                out = model(result, img_cond=img_cond, label_cond=label_cond)
                prob = out[:, :, i, j].softmax(dim = 1)
                result[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)           
                batch_bar.set_postfix({ "pixels": i*size[1]+j})
                batch_bar.update()
    batch_bar.close()
    return result

def train(model, VAE_model, train_loader, config):
    torch.cuda.empty_cache()
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(reduction='mean')
    model.train()
    if GENERATE_SAMPLES:
        VAE_model.eval() # we are not training this
    
    n_epochs = config["n_epochs"]
    best_accuracy = 0.0
    for epoch in range(n_epochs):
        accuracy, train_loss = 0.0, 0.0
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
        # train top
        for i, (data, label, fileid) in enumerate(train_loader):
            img = data[0].to(device)
            if config["top_n_bottom"]: cond = data[1].to(device)
            else: cond=None
            if len(config["conditions"])>0: label_cond = label
            else: label_cond = None
            out = model(img, img_cond=cond, label_cond=label_cond)
            
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
            
            del img
            torch.cuda.empty_cache()
            # debug only
            #if i == 5: break
        
        train_loss /= len(train_loader)
        accuracy /= len(train_loader)
        batch_bar.close() 
        print(f"epoch{epoch}/{n_epochs} acc:{accuracy:.04f}\tloss:{train_loss:.04f}")
        
        info = {
            'train_loss': train_loss,
            'accuracy': accuracy
        }
        if GENERATE_SAMPLES:
            for i, (data, label, fileid) in enumerate(train_loader):
                if config["top_n_bottom"]: img_cond= data[1].to(device)
                else: img_cond=None
                samples = generate_samples(model, 16, config["input_size"], device, img_cond=img_cond, label_cond=label) #LongTensor
                samples = VAE_model.generate(img_cond, samples.detach(), labels=label)
                samples = samples.cpu().numpy()
                samples = np.transpose(samples, (0, 2, 3, 1)) #channel last
                break # run only one generation
            info['samples']=[wandb.Image(
                sample,
                caption=f"gender={label['gender'][i]}, age={label['age'][i].item()}"
            ) for i, sample in enumerate(samples)]
        if USE_WANDB: wandb.log(info)
        
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
    
    
def evaluate(model, val_loader, config):
    model.eval()
    info = {}
    gc.collect()
    torch.cuda.empty_cache()
    
    info['top'], info['labels'], info['bottom']= [], [], []
    info['samples']=[]
    for i, (data, label, fileid) in enumerate(val_loader):
        if config["top_n_bottom"]: t_idxs= data[1].to(device)
        else: t_idxs=None
        b_idxs = generate_samples(model, 8, config["input_size"], device, img_cond=t_idxs, label_cond=label) #LongTensor
        info["labels"], info['top'], info['bottom']= label, t_idxs.detach().clone(), b_idxs.detach().clone()
        break # run only one generation
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    if GENERATE_SAMPLES:
        with open(os.path.join(VAE_MODEL_DIR, "config.json"), "r") as f:
            VAE_config = json.load(f)
        
        if VAE_config["model"] == "single":
            VAE_model = SingleVQVAE(
                3,  #in channels
                VAE_config["hidden_dim"], #hidden dim
                VAE_config["embed_dim"], #embed dim
                VAE_config["n_embed"], #vocab size(dictionary embedding n)
                VAE_config["n_resblocks"], #resblocks inside encoder/decoder,
                conditioned=False
            )
        elif  VAE_config["model"] == "default":
            VAE_model = VQVAE(
                3,  #in channels
                VAE_config["hidden_dim"], #hidden dim
                VAE_config["embed_dim"], #embed dim
                VAE_config["n_embed"], #vocab size(dictionary embedding n)
                VAE_config["n_resblocks"], #resblocks inside encoder/decoder,
                conditioned=len(VAE_config["conditions"])>0
            )

        VAE_model = VAE_model.to(device)
        _, _, specs = utils.load_model(os.path.join(VAE_MODEL_DIR, "epoch_best.pth"), VAE_model)
        VAE_model.eval()
        
        samples = VAE_model.generate(info['top'], info["bottom"], labels=label)
        samples = samples.cpu().numpy()
        samples = np.transpose(samples, (0, 2, 3, 1)) #channel last
        info['samples'] = samples
        
        if USE_WANDB:
            images=[wandb.Image(
                sample,
                caption=f"gender={label['gender'][i]}, age={label['age'][i].item()}"
            ) for i, sample in enumerate(samples)]
            wandb.log({'samples': images})
        else:
            utils.save_numpy_images(samples, SNAIL_MODEL_DIR)
    return info
    
    
if __name__ == "__main__":
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if config["top_n_bottom"]:
        dataset = LmdbDataset(DATA_DIR, labels_dir=LABELS_DIR, keys=["bottom", "top"])
    else:
        dataset = LmdbDataset(DATA_DIR, labels_dir=LABELS_DIR, keys=["top"])
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    device = torch.device("cuda")
    print("training on device: ", device)
    torch.cuda.empty_cache()
    gc.collect()
    
    # pixelsnail for mnist digits = (28, 28), 256 channels for each pixel
    if config["top_n_bottom"]:
        Snailmodel = PixelSnail(config["n_embed"], hidden_dim=config["hidden_dim"], target_size=config["input_size"],
            n_layers = config["n_layers"], 
            n_resblocks = config["n_resblocks"], 
            n_output_layers = config["n_output_layers"], 
            n_cond_img=config["n_cond_img"],
            n_cond_embed=config["n_cond_embed"],
            n_cond_resblocks=3,
            attention=False,
            down_kernel = config["down_kernel"], 
            downright_kernel = config["downright_kernel"], 
            hidden_kernel = config["hidden_kernel"])
    else:
        Snailmodel = PixelSnail(config["n_embed"], hidden_dim=config["hidden_dim"], target_size=config["input_size"],
            n_layers = config["n_layers"], 
            n_resblocks = config["n_resblocks"], 
            n_output_layers = config["n_output_layers"], 
            attention=True,
            down_kernel = config["down_kernel"], 
            downright_kernel = config["downright_kernel"], 
            hidden_kernel = config["hidden_kernel"])
    Snailmodel = nn.DataParallel(Snailmodel)
    #_, _, specs = utils.load_model(os.path.join(SNAIL_MODEL_DIR, "epoch_18.pth"), Snailmodel)
    Snailmodel = Snailmodel.to(device)
    train(Snailmodel, None, train_loader, config)
    #evaluate(Snailmodel, train_loader, config)