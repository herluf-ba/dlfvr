import os
from matplotlib import pyplot as plt
import torch as t
import numpy as np
from torch.utils.data import DataLoader
from svhn import SVHN
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

svhn_train = SVHN(train=True)

for img, label in svhn_train:
    plt.imshow(img.numpy().transpose([1, 2, 0]))
    print(f"{label=}")
    plt.show()
    break

# svhn_train = SVHN(root=os.path.join(os.getcwd(), "..", "datasets"),
# split="train",
# download=True,
# transform=transform)
# svhn_test = SVHN(root=os.path.join(os.getcwd(), "..", "datasets"),
# split="test",
# download=True,
# transform=transform)

# TODO: Use SVHN data set here
train = DataLoader(svhn_train, batch_size=batch_size)
test = DataLoader(svhn_train)

X, y = svhn_train[0]
print(f"{y.shape=}")

model = BaseModel()
opt = t.optim.SGD(model.parameters(), lr=learning_rate,
                  momentum=momentum)  # Stochastic Gradient Descent

fit(model, epochs, loss_func, opt, train, test)

print(f"{model.forward(img).shape=}")
