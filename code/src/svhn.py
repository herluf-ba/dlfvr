import os
import torch
import torch.nn.functional as F
import numpy as np
from display import plot_img, plot_img_vanilla
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Grayscale, Resize, Lambda, ToTensor
from settings import S


class SVHN(Dataset):

    splits = {
        'default_test': {
            'start': 30402,
            'end': 33402
        },
        'default_train': {
            'start': 0,
            'end': 30401
        },
        'develop_train': {
            'start': 0,
            'end': 999
        },
        'develop_test': {
            'start': 1000,
            'end': 1500
        }
    }

    def __init__(self, split, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        assert split in self.splits.keys(), f"No such split '{split}'!"

        data_path = 'train'
        self.img_dir = os.path.join("datasets", data_path)
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, "bbox.csv"))

    def __len__(self):
        partition = self.splits[self.split]
        return partition['end'] - partition['start']

    def __getitem__(self, idx):
        idx += self.splits[self.split]['start']

        # Get and transform image
        image_name = str(idx + 1) + '.png'
        img_path = os.path.join(self.img_dir, image_name)
        image = read_image(img_path)
        transformed_image = self.transform(image) if self.transform else image

        # Get and transform labels
        label_mask = self.img_labels["FileName"] == image_name
        labels = self.img_labels.loc[label_mask].to_numpy()[:, 1:]
        transformed_labels = self.target_transform(
            image.shape[1:], labels) if self.target_transform else labels

        return transformed_image, transformed_labels


def into_resized(original, p):
    return (p[0] / original[0] * 32 * S, p[1] / original[1] * 32 * S)


def squared_distance(p1, p2):
    return (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2


def target_transform(img_size, labels):
    # stfu computer if len(labels) > 4:
    #    print(f"OH NOOOOO there are too many labels: {labels}")

    out = torch.zeros((15, 2, 2))

    # Transform labels into resized space using img size
    unassigned = [(1, *into_resized(img_size, (top, left)),
                   *into_resized(img_size, (height, width)),
                   *(F.one_hot(torch.tensor(c - 1), num_classes=10)).numpy())
                  for (c, left, top, width, height) in labels]

    half_cell = 0.5 / S
    for x in range(S):
        for y in range(S):
            if len(unassigned) > 0:
                # Compute distance to cell center from unassigned labels
                cell_center = into_resized(
                    (1, 1), (x / S + half_cell, y / S + half_cell))

                distances = [
                    squared_distance(cell_center,
                                     (left + width / 2.0, top + height / 2.0))
                    for (_, top, left, height, width, _, _, _, _, _, _, _, _,
                         _, _) in unassigned
                ]

                # Assign the closest one to cell
                closest_i = np.argmin(distances)
                closest = unassigned[closest_i]

                # TODO: There must be a smarter way to do this assignment
                for i in range(15):
                    out[i][x][y] = closest[i]

                del unassigned[closest_i]

    return out


transform = Compose([
    Lambda(lambda img: img / 255),
    Grayscale(),
    Resize((32 * S, 32 * S)),
])


class attribute:
    CONFIDENCE = 0
    LEFT = 1
    TOP = 2
    WIDTH = 3
    HEIGHT = 4
    CLASSES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def batch_extract(tensor_batch, indicies):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    indicies = torch.tensor(indicies).to(device)
    extracted_tensor = torch.index_select(tensor_batch, 1, indicies)
    extracted_tensor = extracted_tensor.reshape(-1, S * S, S*S)
    return extracted_tensor.mT


def batch_extract_confidence(tensor_batch):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    indicies = torch.tensor([attribute.CONFIDENCE]).to(device)
    extracted_confidences = torch.index_select(tensor_batch, 1, indicies)
    extracted_confidences = extracted_confidences.reshape(-1, S * S)
    return extracted_confidences


def batch_extract_bounding_box(tensor_batch):
    return batch_extract(
        tensor_batch,
        [attribute.LEFT, attribute.TOP, attribute.WIDTH, attribute.HEIGHT])


def batch_extract_classes(tensor_batch):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    indicies = torch.tensor(attribute.CLASSES).to(device)
    extracted_classes = torch.index_select(tensor_batch, 1, indicies)
    extracted_classes = extracted_classes.reshape(-1, 10, S * S)
    return extracted_classes.mT
