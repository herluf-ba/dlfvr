import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
import torch.nn.functional as F
from training import fit
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize

from __init__ import BaseModel

batch_size = 64
learning_rate = 0.001
momentum = 0.9
epochs = 10
loss_func = F.mse_loss

# Transform input images
transform = Compose([Grayscale(), Resize((64, 64)), ToTensor()])

svhn_train = SVHN(root="../../datasets",
                  split="train",
                  download=True,
                  transform=transform)
svhn_test = SVHN(root="../../datasets",
                 split="test",
                 download=True,
                 transform=transform)

# TODO: Use SVHN data set here
train = DataLoader(svhn_train, batch_size=batch_size)
test = DataLoader(svhn_test)

model = BaseModel()
opt = t.optim.SGD(model.parameters(), lr=learning_rate,
                  momentum=momentum)  # Stochastic Gradient Descent

fit(model, epochs, loss_func, opt, train, test)

print(f"{model.forward(img).shape=}")
