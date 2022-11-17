import torch as t
import numpy as np
from display import printProgressBar


def loss_batch(model, loss_func, xb, yb, opt=None):
    prediction = model.forward(xb)
    loss = loss_func(prediction, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(model, epochs, loss_func, opt, train_dl, valid_dl, device):
    batch_count = len(train_dl)

    for epoch in range(epochs):
        model.train()
        for batch_i, (xb, yb) in enumerate(train_dl):
            printProgressBar(batch_i,
                             batch_count,
                             prefix=f"Epoch: {epoch} / {epochs}")
            xb, yb = xb.to(device), yb.to(device)
            loss_batch(model, loss_func, xb, yb, opt)
            if (batch_i > 10):
                break

        printProgressBar(batch_count,
                         batch_count,
                         prefix=f"Epoch: {epoch} / {epochs}")

        model.eval()
        with t.no_grad():
            losses, nums = zip(*[
                loss_batch(model, loss_func, xb.to(device), yb.to(device))
                for _, (xb, yb) in enumerate(valid_dl)
            ])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f"validation loss: {val_loss}")
