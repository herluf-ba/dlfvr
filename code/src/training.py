import torch as t
import numpy as np
from metrics import intersection_over_union
from display import printProgressBar


def score_batch(model, loss_func, xb, yb, opt=None):
    prediction = model.forward(xb)
    loss = loss_func(prediction, yb)

    ## compute iou if model is in eval mode
    iou = 0.0
    if not model.training:
        iou = intersection_over_union(prediction, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), iou, len(xb)


def fit(model, epochs, loss_func, opt, train_dl, valid_dl, device):
    # Function in function? Mmmmm delicious spaghetti! 🤌
    def score(dl):
        losses, ious, nums = zip(*[
            score_batch(model, loss_func, xb.to(device), yb.to(device))
            for _, (xb, yb) in enumerate(dl)
        ])

        normalized_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        ## We don't need to compute iou when model is training
        normalized_iou = 0.0
        if not model.training:
            normalized_iou = np.sum(np.multiply(ious, nums)) / np.sum(nums)

        return normalized_loss, normalized_iou

    batch_count = len(train_dl)
    train_loss_hist = []
    val_loss_hist = []
    val_iou_hist = []
    train_iou_hist = []

    for epoch in range(epochs):
        epoch_prefix = f"Epoch: {epoch + 1} / {epochs}"
        model.train()
        for batch_i, (xb, yb) in enumerate(train_dl):
            printProgressBar(batch_i, batch_count, prefix=epoch_prefix)
            xb, yb = xb.to(device), yb.to(device)
            score_batch(model, loss_func, xb, yb, opt)

        printProgressBar(batch_count, batch_count, prefix=epoch_prefix)

        model.eval()
        with t.no_grad():
            print('Calculating validation loss')
            val_loss, val_iou = score(valid_dl)
            print('Calculating training loss ')
            train_loss, train_iou = score(train_dl)

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            val_iou_hist.append(val_iou)
            train_iou_hist.append(train_iou)

            print(f"{val_loss=} {train_loss=} {val_iou=} {train_iou=}")

    return train_loss_hist, val_loss_hist
