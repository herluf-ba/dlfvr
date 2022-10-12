# Structure inspired by: https://pytorch.org/tutorials/beginner/nn_tutorial.html#nn-sequential 
# Pad layer inspiration: https://stackoverflow.com/questions/62166719/padding-same-conversion-to-pytorch-padding 
# Why do we pad the maxPool2d operation with same? 
import torch as t; 
from torch import Tensor 
import torch.nn as nn 

class BasisModel(nn.Module): 
    def __init__(self): 
        super().__init__()

        # The encoder used for all 3 operations
        self.encoded = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=33, padding="same"), 
            nn.ReLU(), 
            # Add a Pad layer here to preserve dimensions 
            nn.MaxPool2d((2,2)), 

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size = 3), 
            nn.ReLU(), 
            # Add a Pad layer here to preserve dimensions 
            nn.MaxPool2d((2,2)), 
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3), 
            nn.ReLU(), 
            # Add a Pad layer here to preserve dimensions 
            nn.MaxPool2d((2,2))
        )


        # Whether there is an object in a (4x4?) cell or not (all of the colored squares)
        # I think in_channels for the first should be 32, as this is the output from the encoder. 
        self.confidence = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            # Add a Pad layer here to preserve dimensions 
            nn.MaxPool2d((2,2)), 
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding="same"), 
            nn.ReLU(),
            # Add pad layer here to preserve dimensions
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels = 16, out_channels=1, kernel_size=1, padding="same"), 
            nn.Sigmoid() 
        ); 

        # Bounding boxes predicts the bounding box coordinates of each cell
        self.bounding_box = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, padding="same"),
            nn.ReLU(),
            # Add a padding layer here to preserve dimensions
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding="same"),
            nn.ReLU(),
            # Add a padding layer here to preserve dimensions
            nn.MaxPool2d((2,2)), 
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding="same")

        );

        # Classes predicts probability of each class being in a (4x4?) cell 
        self.classes = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding="same"), 
            nn.ReLU(), 
            # Add a padding layer here
            nn.MaxPool2d((2,2)), 
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding="same"), 
            nn.ReLU(), 
            # Add a padding layer here
            nn.MaxPool2d((2,2)), 
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, padding="same"),
            nn.Softmax(dim=0) # Commit softmax over depth dimension (it's the third in keras, first in pytorch)
        ) 

    def forward(self, x): 
        encoded = self.encoded(x)
        confidence = self.confidence(encoded)
        bounding_box = self.bounding_box(encoded)
        classes = self.classes(encoded)
        return (encoded, confidence, bounding_box, classes)

model = BasisModel(); 

#format: img[channel][row][column]
dims = 64 
img = t.rand((1,dims,dims)); 

img_encoded, img_confidence, img_bounding_box, img_classes = model.forward(img)

print(f"🍆{img.shape=} {img_encoded.shape=} {img_confidence.shape=} {img_bounding_box.shape=} {img_classes.shape=}")
