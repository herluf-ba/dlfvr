import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class SVHN(Dataset):

    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        if not train:
            print("JK we aint got no test data")
            exit()

        self.img_dir = os.path.join("datasets",
                                    "train") if train else os.path.join(
                                        "datasets", "test")

        self.img_labels = pd.read_csv(os.path.join(self.img_dir, "bbox.csv"))

    def __len__(self):
        return len(self.img_labels)

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
        print(f"{image / 255}")

        return image, labels
