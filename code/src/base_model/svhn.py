import os
from display import plot_img, plot_img_vanilla
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class SVHN(Dataset):

    splits = {
            'test': {
                'start': 30402, 
                'end': 33402
                }, 
            'train': {
                'start': 0, 
                'end': 30401
                }, 
            'develop_train': {
                'start': 0, 
                'end': 999 
                },
            'develop_test': {
                'start': 100, 
                'end': 1500
                }
            }

    def __init__(self, split="train", transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        
        assert split in self.splits.keys(), f"ERROR ERROR ERROR! NO SUCH SPLIT '{split}'. Fuck you."

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
