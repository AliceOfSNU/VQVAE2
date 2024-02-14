from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import os

DATA_DIR = 'VQVAE/data/ffhq_images/images_original'
SAVE_DIR = 'VQVAE/data/ffhq_images/images_resized_05000'
def crop_images():
    for path_dir in os.listdir(DATA_DIR):
        if not os.path.isdir(os.path.join(DATA_DIR, path_dir)):continue
        path_dir = os.path.join(DATA_DIR, path_dir)
        print("processing ", path_dir)
        for file in os.listdir(path_dir):
            outfile = os.path.join(SAVE_DIR, file)
            img = Image.open(os.path.join(path_dir, file))
            img = img.resize((256, 256))
            img.save(outfile)

    
crop_images()