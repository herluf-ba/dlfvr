import torch as t
import torch.nn as nn
''' 
TODO: 
    - Why padding="same" for max pooling operation in keras? 
      - If the above has a good answer: Find a way to pad pooling operations
        (Have a look at: https://stackoverflow.com/questions/62166719/padding-same-conversion-to-pytorch-padding)
'''


# Inspired by: https://pytorch.org/tutorials/beginner/nn_tutorial.html#nn-sequential
class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoded = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=33,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=3,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)))

        # Confidence: is one of the 4x4 cells an object?
        self.confidence = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=4,
                      kernel_size=3,
                      stride=1,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=4,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=16,
                      out_channels=1,
                      kernel_size=1,
                      padding="same"), nn.Sigmoid())

        # bounding_box: for each cell find bounding coordinates for element if present
        self.bounding_box = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=4,
                      kernel_size=3,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=4,
                      out_channels=4,
                      kernel_size=3,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=4,
                      out_channels=4,
                      kernel_size=1,
                      padding="same"))

        # classes: for each cell predict class present in cell
        self.classes = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=16,
                      out_channels=10,
                      kernel_size=1,
                      padding="same"),
            # TODO: A bit in doubt whether dim is correct
            nn.Softmax(dim=0))

    def forward(self, x):
        enc = self.encoded(x)

        return t.cat(
            (self.confidence(enc), self.bounding_box(enc), self.classes(enc)),
            dim=1)
