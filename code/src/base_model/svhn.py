import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


def plot_img_vanilla(image, labels):
    fig, ax = plt.subplots()
    ax.imshow(image.numpy().transpose([1, 2, 0]), cmap='gray')

    for (_, left, top, width, height) in labels:
        ax.add_patch(
            patches.Rectangle((left, top),
                              width,
                              height,
                              linewidth=1,
                              edgecolor='r',
                              facecolor='none'))
    plt.show()


def plot_img(image, labels):
    fig, ax = plt.subplots()
    ax.imshow(image.numpy().transpose([1, 2, 0]), cmap='gray')

    for x in range(labels.shape[1]):
        for y in range(labels.shape[2]):
            top = labels[1][x][y]
            left = labels[2][x][y]
            width = labels[4][x][y]
            height = labels[3][x][y]
            ax.add_patch(
                patches.Rectangle((left, top),
                                  width,
                                  height,
                                  linewidth=1,
                                  edgecolor='r',
                                  facecolor='none'))
    plt.show()


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

        if idx > 2:
            exit()

        print(f"{idx=}")

        plot_img_vanilla(image, labels)
        plot_img(transformed_image, transformed_labels)

        return transformed_image, transformed_labels
