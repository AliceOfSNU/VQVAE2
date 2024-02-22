from PIL import Image
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
import os
import lmdb
import json

FFHQ_DATA_DIR = 'VQVAE/data/ffhq_images'
FFHQ_LABELS_DIR = 'VQVAE/data/ffhq-features-dataset-master'
CAT_DATA_DIR = 'VQVAE/data/cat_faces/cats'

class FFHQDataset(Dataset):
    def __init__(self, data_dir = FFHQ_DATA_DIR, labels_dir = FFHQ_LABELS_DIR):
        # helper function
        def flatten_dict(dic, pfx=""):
            add_dict = {}
            for k, v in dic.items():
                if isinstance(v, dict):
                    add_dict.update(flatten_dict(v, k))
                else:
                    k = k if pfx == "" else pfx+"_"+k
                    add_dict[k] = v
            return add_dict
        self.img_files = []
        print("create memory efficient dataset from data ", data_dir)
        self.labels = np.load(f'{labels_dir}/all_labels.npy',allow_pickle=True).item()
        print("all labels loaded.. ")
        all_cnt = 0
        for path_dir in os.listdir(data_dir):
            if not os.path.isdir(os.path.join(data_dir, path_dir)):continue
            if "resized" not in path_dir: continue
            path_dir = os.path.join(data_dir, path_dir)
            print("processing ", path_dir)
            for file in os.listdir(path_dir):
                if file.split('.')[0] not in self.labels: continue #check label exists
                if all_cnt >= 100: break
                self.img_files.append(os.path.join(path_dir, file))
                all_cnt += 1
        
        print("\ttotal image cnt: ", len(self.img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, ind):
        data = Image.open(self.img_files[ind])
        data = transforms.ToTensor()(data)
        file_id = self.img_files[ind].split('/')[-1].split('.')[0]
        return data, self.labels[file_id]


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