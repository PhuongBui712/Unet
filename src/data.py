import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CarvanaDataset(Dataset):
    SIZE = (512, 512)

    def __init__(self, path, split):
        if split == 'test':
            # TODO: handle test set
            pass
        else:
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            self.images = [data_dir + '/train_hq/' + file for file in os.listdir(data_dir + '/train_hq/')]
            self.masks = [data_dir + '/train_masks/' + file for file in os.listdir(data_dir + '/train_masks/')]

        self.transformer = transforms.Compose([
            transforms.Resize(size=self.SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        mask = Image.open(self.masks[index]).convert('L')

        return self.transformer(image), self.transformer(mask)
    

class BCSSDataset(Dataset):
    SIZE=(224, 224)
    _img_transformer = transforms.Compose([
            transforms.Resize(SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    _mask_transformer = transforms.Compose([
            transforms.Resize(SIZE),
            transforms.PILToTensor(),
        ])
    
    def __init__(self, image_path: str, mask_path: str):
        image_path = os.path.abspath(image_path)
        mask_path = os.path.abspath(mask_path)
        
        self.images = [os.path.join(image_path, filename) for filename in os.listdir(image_path)]
        self.masks = [os.path.join(mask_path, filename) for filename in os.listdir(mask_path)]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        image = self._img_transformer(image)

        mask = Image.open(self.masks[idx])
        mask = self._mask_transformer(mask)
        mask = torch.squeeze(mask, 0).long()

        return image, mask
