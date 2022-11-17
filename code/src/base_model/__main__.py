import torch as t
from torch.utils.data import DataLoader
from svhn import SVHN
import torch.nn.functional as F
from training import fit
from torchvision.transforms import Compose, Grayscale, Resize

from __init__ import BaseModel

batch_size = 64
learning_rate = 0.001
momentum = 0.9
epochs = 10
loss_func = F.mse_loss

S = 2
# Transform input images
transform = Compose([
    Grayscale(),
    Resize((32 * S, 32 * S)),
])


def target_transform(_labels):
    return t.rand((15, 2, 2))


svhn_train = SVHN(split='train',
                  transform=transform,
                  target_transform=target_transform)
svhn_dev = SVHN(split='dev',
                transform=transform,
                target_transform=target_transform)

train = DataLoader(svhn_train, batch_size=batch_size)
dev = DataLoader(svhn_dev)

img, y = svhn_train[0]

model = BaseModel()
opt = t.optim.SGD(model.parameters(), lr=learning_rate,
                  momentum=momentum)  # Stochastic Gradient Descent

fit(model, epochs, loss_func, opt, train, dev)

# print(f"{model.forward().shape=}")
