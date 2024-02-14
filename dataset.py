from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os

FFHQ_DATA_DIR = 'VQVAE/data/ffhq_images'
CAT_DATA_DIR = 'VQVAE/data/cat_faces/cats'

class FFHQDataset(Dataset):
    def __init__(self, data_dir = FFHQ_DATA_DIR):
        self.img_files = []
        print("create memory efficient dataset from data ", data_dir)
        for path_dir in os.listdir(data_dir):
            if not os.path.isdir(os.path.join(data_dir, path_dir)):continue
            if "resized" not in path_dir: continue
            path_dir = os.path.join(data_dir, path_dir)
            print("processing ", path_dir)
            for file in os.listdir(path_dir):
                self.img_files.append(os.path.join(path_dir, file))
        
        print("\ttotal image cnt: ", len(self.img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, ind):
        data = Image.open(self.img_files[ind])
        data = transforms.ToTensor()(data)
        return data


class CatsDataset(Dataset):
    def __init__(self, data_dir = CAT_DATA_DIR):
        self.img_files = []
        print("create memory efficient dataset from data ", data_dir)
        for path_dir in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, path_dir)):continue
            path_dir = os.path.join(data_dir, path_dir)
            self.img_files.append(path_dir)
        
        print("\ttotal image cnt: ", len(self.img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, ind):
        data = Image.open(self.img_files[ind])
        data = transforms.ToTensor()(data)
        return data
