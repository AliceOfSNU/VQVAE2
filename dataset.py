from PIL import Image
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
import os
import lmdb
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
        self.data_dir = data_dir
        print("create memory efficient dataset from data ", data_dir)
        for path_dir in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, path_dir)):continue
            self.img_files.append(path_dir)
        
        print("\ttotal image cnt: ", len(self.img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, ind):
        path_dir = os.path.join(self.data_dir, self.img_files[ind])
        data = Image.open(path_dir)
        data = transforms.ToTensor()(data)
        return data, self.img_files[ind]

class LmdbDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        print("create lmdb dataset from data ", data_dir)
        self.db = lmdb.open(
            data_dir,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with self.db.begin(write=False) as txn:
            self.length = int(txn.get("length".encode('utf-8')).decode('utf-8'))
        print(f"\t contains {self.length}rows.")
    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        with self.db.begin(write=False) as txn:
            data = pickle.loads(txn.get(str(ind).encode('utf-8')))
        return torch.from_numpy(data["code"]), data["filename"]