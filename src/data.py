import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


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
    def __init__(self, path: str, split: Literal['train', 'val', 'test'] = 'train'):
        path = os.path.abspath(path)
        image_path = os.path.join(path, split)
        self.images = [os.path.join(image_path, filename) for filename in os.listdir(image_path)]
        mask_path = os.path.join(path, f'{split}_mask')
        self.masks = [os.path.join(mask_path, filename) for filename in os.listdir(mask_path)]

        self.transformer = transforms.Compose([
            transforms.Resize(self.SIZE),
            transforms.ToTensor()
        ])

        # TODO: handle test set

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx]).convert('L')

        return self.transformer(image), self.transformer(mask)
