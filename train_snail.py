from pixelsnail import PixelSnail
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

train_config = {
    "n_epochs" :3,
    "lr" :1e-3,
    "hidden_dim": 128,
    "n_layers": 2,
    "n_resblocks": 4,
    "batch_size": 32,
    "down_kernel": (2, 5),
    "downright_kernel":(2, 3),
    "hidden_kernel":(3, 5),
    "run_id": "initial_setup",
    "seed": 12,
}


BASE_DIR = "VQVAE"
MODEL_DIR = os.path.join(BASE_DIR, "model")

np.random.seed(train_config["seed"])
torch.manual_seed(train_config["seed"])

torch.cuda.empty_cache()
gc.collect()


# mnist dataset
mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
])
dataset = datasets.MNIST('./data', transform=mnist_transform, download=True)
train_loader = DataLoader(dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=1)

# pixelsnail for mnist digits = (28, 28), 256 channels for each pixel
model = PixelSnail(256, hidden_dim=train_config["hidden_dim"], target_size=(28, 28),
                   n_layers = train_config["n_layers"], n_resblocks=train_config["n_resblocks"], n_output_layers=1, 
                   down_kernel = train_config["down_kernel"], 
                   downright_kernel=train_config["downright_kernel"], 
                   hidden_kernel=train_config["hidden_kernel"])

device = torch.device("cuda")
model = model.to(device)
print("training on device: ", device)

if USE_WANDB:
    run = wandb.init(
        name = train_config["run_id"], 
        reinit = True, 
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "PixelCNN_VQVAE", 
        config=train_config
    )

def generate_samples(model, N, size, device):
    # return 1, H, W generated samples from model.
    model.eval()
    result = torch.zeros(N, *size, dtype=torch.int64).to(device)
    for i in range(size[0]):
        for j in range(size[1]):
            out = model(result)
            prob = out[:, :, i, j].softmax(dim = 1)
            result[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)
            
    model.train()
    # add channel dimension =1
    result = result.unsqueeze(1)
    return result.cpu().detach().numpy()

def train(model, train_loader, train_config):
    optimizer = optim.Adam(model.parameters(), lr=train_config["lr"])
    criterion = nn.CrossEntropyLoss(reduction='mean')
    model.train()
    
    n_epochs = train_config["n_epochs"]
    best_accuracy = 0.0
    for epoch in range(n_epochs):
        accuracy, train_loss = 0.0, 0.0
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
        for i, (img, label) in enumerate(train_loader):
            img = (img.to(device)*255.).long().squeeze(1) #B, 28, 28 
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
        
        samples = generate_samples(model, 10, (28, 28), device)
        samples = np.transpose(samples, (0, 2, 3, 1))
        if USE_WANDB:
            examples = pred[:10].unsqueeze(-1).detach().cpu().numpy()
            gt = img[:10].unsqueeze(-1).detach().cpu().numpy()
            wandb.log({
                'train_loss': train_loss,
                'reconstr_img': [wandb.Image(ex) for ex in examples],
                'gt_img': [wandb.Image(ex) for ex in gt],
                'samples': [wandb.Image(sample) for sample in samples],
            })
            
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            utils.save_model(MODEL_DIR, epoch, model, optimizer, stats_dict={
                "accuracy": accuracy,
                "config": train_config
            })
    
train(model, train_loader, train_config)
if USE_WANDB:
    run.finish()

