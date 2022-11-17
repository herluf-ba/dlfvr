import os
from display import plot_img, plot_img_vanilla
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
        # TODO: find highest number of image, not label count
        return 10 if self.split == 'dev' else len(self.img_labels)

    def __getitem__(self, idx):
        # Get and transform image
        image_name = str(idx + 1) + '.png'
        img_path = os.path.join(self.img_dir, image_name)
        image = read_image(img_path) / 255
        transformed_image = self.transform(image) if self.transform else image

        # Get and transform labels
        label_mask = self.img_labels["FileName"] == image_name
        labels = self.img_labels.loc[label_mask].to_numpy()[:, 1:]
        transformed_labels = self.target_transform(
            image.shape[1:], labels) if self.target_transform else labels

        # print(f"{idx=}")
        # plot_img_vanilla(image, labels)
        # plot_img(transformed_image, transformed_labels)

        return transformed_image, transformed_labels
