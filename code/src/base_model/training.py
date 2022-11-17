import torch as t
import numpy as np


# Print iterations progress
def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=100,
                     fill='━',
                     printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def loss_batch(model, loss_func, xb, yb, opt=None):
    prediction = model.forward(xb)
    loss = loss_func(prediction, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(model, epochs, loss_func, opt, train_dl, valid_dl):
    batch_count = len(train_dl)

    for epoch in range(epochs):
        model.train()
        for batch_i, (xb, yb) in enumerate(train_dl):
            printProgressBar(batch_i,
                             batch_count,
                             prefix=f"Epoch: {epoch} / {epochs}")
            loss_batch(model, loss_func, xb, yb, opt)
            if (batch_i > 10):
                break

        model.eval()
        with t.no_grad():
            losses, nums = zip(*[
                loss_batch(model, loss_func, xb, yb)
                for _, (xb, yb) in enumerate(valid_dl)
            ])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)
