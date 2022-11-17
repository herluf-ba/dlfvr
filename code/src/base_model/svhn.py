import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class SVHN(Dataset):

    def __init__(self, split="train", transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        if split == 'test':
            print("JK we aint got no test data")
            exit()

        data_path = 'train' if split == 'dev' else split
        self.img_dir = os.path.join("datasets", data_path)
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, "bbox.csv"))

    def __len__(self):
        return 10 if self.split == 'dev' else len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path) / 255
        labels = self.img_labels.loc[self.img_labels["FileName"] == str(idx +
                                                                        1) +
                                     ".png"].to_numpy()[:, 1:]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        return image, labels
