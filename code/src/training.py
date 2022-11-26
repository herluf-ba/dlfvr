import torch as t
import numpy as np
from display import printProgressBar
from svhn import batch_extract_classes


def loss_batch(model, loss_func, xb, yb, opt=None, print_prediction=False):
    prediction = model.forward(xb)
    loss = loss_func(prediction, yb)

    if(print_prediction): 
        print(batch_extract_classes(prediction)[0])
        print(batch_extract_classes(yb)[0])

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(model, epochs, loss_func, opt, train_dl, valid_dl, device):

    # Function in function? Mmmmm delicious spaghetti!
    def score(dl):
        losses, nums = zip(*[
            loss_batch(model, loss_func, xb.to(device), yb.to(device))
            for _, (xb, yb) in enumerate(dl)
            ])

        return np.sum(np.multiply(losses, nums)) / np.sum(nums)

    batch_count = len(train_dl)
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(epochs):
        epoch_prefix = f"Epoch: {epoch + 1} / {epochs}"
        model.train()
        for batch_i, (xb, yb) in enumerate(train_dl):
            printProgressBar(batch_i,
                             batch_count,
                             prefix=epoch_prefix)
            xb, yb = xb.to(device), yb.to(device)
            is_last_loss_batch = batch_i == len(train_dl) - 1 and epoch == epochs - 1
            loss_batch(model, loss_func, xb, yb, opt, print_prediction=is_last_loss_batch)

        printProgressBar(batch_count,
                         batch_count,
                         prefix=epoch_prefix)

        model.eval()
        with t.no_grad():
            print('Calculating validation loss')
            val_loss= score(valid_dl)
            print('Calculating training loss ')
            train_loss = score(train_dl)

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
        print(f"{val_loss=} {train_loss=}")

    return train_loss_hist, val_loss_hist
