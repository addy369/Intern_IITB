
from torchvision import utils,transforms
import torch
import random
import glob
import os
from PIL import Image



class train2(torch.utils.data.Dataset):
    def __init__(self, root, train = True):
        self.root = root
        self.train = train

        # self.out = self.domain_label
        self.samples = []
        dirs = glob.glob(root +'/*') # list cancer, non-cancer
        for i, item in enumerate(dirs):
        	items = os.listdir(item)
        	for file_ in items:
        		self.samples.append([os.path.join(item, file_), i])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img, label = self.samples[index]
        # file_location,label, domain_output,root_folder = sample
        transformations = transforms.Compose([transforms.ToTensor()])
        # array = []
        img = Image.open(img)
        img = transformations(img).float()
        return img,label
