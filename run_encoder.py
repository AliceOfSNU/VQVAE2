import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import gc
from tqdm import tqdm
import numpy as np
import os
import json
import argparse
from collections import namedtuple
import lmdb
import pickle

from vqvae import VQVAE, SingleVQVAE
from dataset import FFHQDataset, CatsDataset
import utils

BASE_DIR = "VQVAE"
FFHQ_DATA_DIR = os.path.join(BASE_DIR, 'data/ffhq_images')
CATS_DATA_DIR = os.path.join(BASE_DIR, 'data/cat_faces/cats')
MODEL_DIR = os.path.join(BASE_DIR, "model")

# example config to show its contents
config = {
    "n_epochs" :5,
    "lr" :1e-3,
    "hidden_dim": 128,
    "embed_dim": 64,
    "n_embed": 128,
    "n_resblocks": 2,
    "seed": 12,
    "batch_size": 32,
    "latent_loss_weight":0.2,
    "model": "default"
}

## contains code from rosnality/vq-vae-2-pytorch

def extract(lmdb_env, loader, model, device, config):
    #if config["model"] == "default":
    #    CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])
    #elif config["model"] == "single":
    #    CodeRow = namedtuple('CodeRow', ['code', 'filename'])

    index = 0
    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for data in pbar:
            # run model encode and write result for each image
            if config["model"] == "default":
                img = data[0].to(device)
                filename = data[2]
                _, _, _, id_t, id_b = model.encode(img)
                id_t = id_t.detach().cpu().numpy()
                id_b = id_b.detach().cpu().numpy()
                for file, top, bottom in zip(filename, id_t, id_b):
                    row = {"top": top, "bottom": bottom, "filename": file}
                    txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                    index += 1
                    pbar.set_description(f'inserted: {index}')
            elif config["model"] == "single":
                img = data[0].to(device)
                filename = data[1]
                _, _, idxs = model.encode(img)
                idxs = idxs.detach().cpu().numpy()
                for file, idx in zip(filename, idxs):
                    row = {"code":idx, "filename":file}
                    txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                    index += 1
                    pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str)
    
    args = parser.parse_args()
    # loads json config
    load_path = os.path.join(MODEL_DIR, args.ckpt)
    with open(os.path.join(load_path, "config.json"), "r") as f:
        config = json.load(f)
    run_id = config["run_id"]
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    torch.cuda.empty_cache()
    gc.collect()

    # create model
    if config["model"] == "default":
        train_data = FFHQDataset(data_dir=FFHQ_DATA_DIR)
        train_loader = DataLoader(train_data, batch_size=64, num_workers=1)
        model = VQVAE(
            3,  #in channels
            config["hidden_dim"], #hidden dim
            config["embed_dim"], #embed dim
            config["n_embed"], #vocab size(dictionary embedding n)
            config["n_resblocks"] #resblocks inside encoder/decoder
        )
    elif config["model"] == "single":
        train_data = CatsDataset(data_dir=CATS_DATA_DIR)
        train_loader = DataLoader(train_data, batch_size=64, num_workers=1)
        model = SingleVQVAE(
            3,  #in channels
            config["hidden_dim"], #hidden dim
            config["embed_dim"], #embed dim
            config["n_embed"], #vocab size(dictionary embedding n)
            config["n_resblocks"] #resblocks inside encoder/decoder
        )

    # load weights
    model, _, specs = utils.load_model(os.path.join(load_path, "epoch_best.pth"), model)
    
    # device
    device = torch.device("cuda")
    model = model.to(device)
    print("running eval on device: ", device)
    model.eval()

    # code from rosanality/vq-vae-2-pytorch
    map_size = 100 << 20
    save_path = os.path.join(load_path, "code.lmdb")
    env = lmdb.open(save_path, map_size=map_size)
    extract(env, train_loader, model, device, config)
    
    
