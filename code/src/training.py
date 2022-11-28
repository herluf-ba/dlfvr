import torch as t
import torch.nn as nn
import inspect
import numpy as np
from display import printProgressBar


def score_batch(model, loss_func, xb, yb, opt=None, logger=None):
    prediction = model.forward(xb)

    should_log = logger and 'logger' in inspect.getargspec(loss_func).args
    loss = loss_func(prediction, yb,
                     logger=logger) if should_log else loss_func(
                         prediction, yb)

    if opt is not None:
        #for param_tensor in model.state_dict():
        #    print(param_tensor, "\t", model.state_dict()[param_tensor].grad)

        loss.backward()
        print(f'{model.encoded[0].weight.grad.mean()=}')
        print(f'{model.bounding_box[-1].weight.grad.mean()=}')
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(model, epochs, loss_func, opt, train_dl, valid_dl, device, logger):
    # Function in function? Mmmmm delicious spaghetti! 🤌
    def score(dl):
        ## Compute loss
        losses, nums = zip(*[
            score_batch(
                model, loss_func, xb.to(device), yb.to(device), logger=logger)
            for _, (xb, yb) in enumerate(dl)
        ])

        ## Save epoch log data to logger history
        logger.commit_epoch()

    batch_count = len(train_dl)

    for epoch in range(epochs):
        epoch_prefix = f"Epoch: {epoch + 1} / {epochs}"
        model.train()
        for batch_i, (xb, yb) in enumerate(train_dl):
            printProgressBar(batch_i, batch_count, prefix=epoch_prefix)
            xb, yb = xb.to(device), yb.to(device)
            score_batch(model, loss_func, xb, yb, opt, logger=None)

        printProgressBar(batch_count, batch_count, prefix=epoch_prefix)

        model.eval()
        with t.no_grad():

            print('Calculating validation loss')
            logger.set_mode('val')
            score(valid_dl)

            print('Calculating training loss ')
            logger.set_mode('train')
            score(train_dl)

        

