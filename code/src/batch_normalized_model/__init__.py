import torch as t
import torch.nn as nn

# Inspired by: https://pytorch.org/tutorials/beginner/nn_tutorial.html#nn-sequential
class BatchNormalizedModel(nn.Module):

    def __init__(self, weight_init=None):
        super().__init__()

        self.encoded = nn.Sequential(
            nn.BatchNorm2d(1), 
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=33,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(8), 
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=3,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(16), 
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(32))

        self.confidence = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=4,
                      kernel_size=3,
                      stride=1,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(16), 
            nn.Conv2d(in_channels=16,
                      out_channels=1,
                      kernel_size=1,
                      padding="same"), 
            ) #TODO: it was originally: nn.Sigmoid()

        # bounding_box: for each cell find bounding coordinates for element if present
        self.bounding_box = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=4,
                      kernel_size=3,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4,
                      out_channels=4,
                      kernel_size=3,
                      padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(4),
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
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(16), 
            nn.Conv2d(in_channels=16,
                      out_channels=10,
                      kernel_size=1,
                      padding="same"),
                      #nn.Softmax(dim=1) # TODO: Find a way to softmax here. We need it when doing predictions
                      ) 
        
        # Initialize weights 
        if (weight_init is not None):
            layers_initialized = 0
            assert weight_init in ['kaiming_uniform', 'kaiming_normal'], f'Unknown weight init {weight_init}'
            models = [self.encoded, self.classes, self.confidence, self.bounding_box]
            for model in models:
                for layer in model:
                    is_conv_layer = isinstance(layer, nn.Conv2d)
                    if is_conv_layer and weight_init == 'kaiming_uniform': 
                            layers_initialized += 1
                            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    elif is_conv_layer and weight_init == 'kaiming_normal': 
                            layers_initialized += 1
                            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

            print(f'Initialized {layers_initialized} layers using {weight_init}')

    def forward(self, x):
        enc = self.encoded(x)

        return t.cat(
            (self.confidence(enc), self.bounding_box(enc), self.classes(enc)),
            dim=1)
