# Structure inspired by: https://pytorch.org/tutorials/beginner/nn_tutorial.html#nn-sequential 
# Pad layer inspiration: https://stackoverflow.com/questions/62166719/padding-same-conversion-to-pytorch-padding 
import torch.nn as nn 

class BasisModel(nn.Module): 
    def __init__(self): 
        super().__init__()

        # The encoder used for all 3 operations
        encoded = nn.Sequential(
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
        confidence = nn.Sequential(
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
        box = None 

        # Classes predicts probability of each class being in a (4x4?) cell 

model = BasisModel(); 
