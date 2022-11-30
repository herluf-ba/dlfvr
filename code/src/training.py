import torch as t
import torch.nn as nn
import inspect
import numpy as np
from display import printProgressBar, bcolors
from settings import batch_extract_classes
from inspection import plot_grad_flow


def score_batch(model,
                loss_func,
                xb,
                yb,
                opt=None,
                logger=None,
                compute_metrics=False,
                should_collect_model_data=False):
    prediction = model.forward(xb)
    loss = loss_func(prediction,
                     yb,
                     logger=logger if compute_metrics else None)

    if opt is not None:
        loss.backward()
        if should_collect_model_data:
            # Collect weights and gradients for all weight layers right after each backward pass
            logger.collect_model_data()

        opt.step()
        opt.zero_grad()

    if compute_metrics:
        # Compute accuracy for classes
        target_classes = batch_extract_classes(yb).detach().numpy()
        pred_classes = batch_extract_classes(prediction).detach().numpy()
        true_positives = target_classes.argmax(axis=2) == pred_classes.argmax(
            axis=2)
        accuracy = true_positives.mean()
        logger.add_metric("accuracy", accuracy)

        ## TODO Compute accuracy for confidence

    return loss.item(), len(xb)


def fit(model, epochs, loss_func, opt, train_dl, valid_dl, device, logger):
    batch_count = len(train_dl)
    batch_count_val = len(valid_dl)
    for epoch in range(epochs):
        epoch_prefix = f"Epoch: {epoch + 1} / {epochs}"
        model.train()
        for batch_i, (xb, yb) in enumerate(train_dl):
            printProgressBar(batch_i, batch_count, prefix=epoch_prefix)
            xb, yb = xb.to(device), yb.to(device)
            score_batch(model,
                        loss_func,
                        xb,
                        yb,
                        opt,
                        logger=logger,
                        should_collect_model_data=True)

        printProgressBar(batch_count, batch_count, prefix=epoch_prefix)

        model.eval()
        with t.no_grad():
            #print('Calculating loss')
            logger.set_mode('val')
            for b, (xb, yb) in enumerate(valid_dl):
                printProgressBar(b, batch_count_val, prefix='Val loss ', barColor=bcolors.OKCYAN)
                score_batch(model, loss_func, xb.to(device), yb.to(device), logger=logger, compute_metrics=True)
            printProgressBar(batch_count_val, batch_count_val, prefix='Val loss ', barColor=bcolors.OKCYAN)
            
            logger.set_mode('train')
            for b, (xb, yb) in enumerate(train_dl):
                printProgressBar(b, batch_count, prefix='Train loss ', barColor=bcolors.WARNING)
                score_batch(model, loss_func, xb.to(device), yb.to(device), logger=logger, compute_metrics=True)
            printProgressBar(batch_count, batch_count, prefix='Train loss ', barColor=bcolors.WARNING)

            ## Save epoch log data to logger history
            logger.commit_epoch()
