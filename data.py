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
