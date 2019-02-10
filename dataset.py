import torch 
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import ast

class MemeDataset(Dataset):
    def __init__(self,img_paths,caplens,enc_caps, transform=None):
        self.img_paths = np.load(img_paths)
        self.caplens = np.load(caplens)
        self.enc_caps = np.load(enc_caps)
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self,idx):
        img_name = self.img_paths[idx]
        image = Image.open(img_name)

        caption = torch.tensor(self.enc_caps[idx])
        caption_len = torch.tensor(self.caplens[idx])
        # print(caption.shape,caption_len.shape)      
        if self.transform:
            image = self.transform(image)
        return (image,caption,caption_len)
    