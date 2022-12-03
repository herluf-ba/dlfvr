import torch as t
import torch.nn as nn
import inspect
import numpy as np
from display import printProgressBar, bcolors
from settings import CONFIDENCE_THRESHOLD, batch_extract_classes, batch_extract_confidence, batch_extract_bounding_box

def score_batch(model,
                loss_func,
                xb,
                yb,
                opt=None,
                logger=None,
                compute_metrics=False,
                collect_gradients=False):

    prediction = model.forward(xb)
    loss = loss_func(prediction,
                     yb,
                     logger=logger if compute_metrics else None)

    if opt is not None:
        loss.backward()
        if collect_gradients:
            logger.collect_gradients()
        opt.step()
        opt.zero_grad()

    if compute_metrics:
        t_conf = batch_extract_confidence(yb)
        t_classes = batch_extract_classes(yb)
        t_bounding_boxes = batch_extract_bounding_box(yb)

        p_conf = batch_extract_confidence(prediction)
        p_classes = batch_extract_classes(prediction)
        p_bounding_boxes = batch_extract_bounding_box(prediction)
        conf_filter = t_conf > 0
        
        logger.collect_classes_metrics(t_classes[conf_filter], p_classes[conf_filter])
        logger.collect_confidence_metrics(t_conf, p_conf > CONFIDENCE_THRESHOLD)
        #logger.collect_bounding_box_metrics(t_bounding_boxes, p_bounding_boxes)

    return loss.item(), len(xb)


def fit(model, epochs, loss_func, opt, train_dl, valid_dl, device, logger, lr_scheduler=None):
    batch_count = len(train_dl)
    batch_count_val = len(valid_dl)

    for epoch in range(epochs):
        epoch_prefix = f"Epoch: {epoch + 1} / {epochs}"
        ## TRAINING
        model.train()
        for batch_i, (xb, yb) in enumerate(train_dl):
            printProgressBar(batch_i, batch_count, prefix=epoch_prefix)
            score_batch(model, loss_func, xb.to(device), yb.to(device), opt, logger=logger, collect_gradients=True)
        printProgressBar(batch_count, batch_count, prefix=epoch_prefix)
        
        # Step LR scheduler at end of epoch
        if lr_scheduler is not None: 
            lr_scheduler.step()

        ## CALCULATION LOSS
        model.eval()
        with t.no_grad():
            # VALIDATION LOSS
            logger.set_mode('val')
            for b, (xb, yb) in enumerate(valid_dl):
                printProgressBar(b, batch_count_val, prefix='Val loss ', barColor=bcolors.OKCYAN)
                score_batch(model, loss_func, xb.to(device), yb.to(device), logger=logger, compute_metrics=True)
            printProgressBar(batch_count_val, batch_count_val, prefix='Val loss ', barColor=bcolors.OKCYAN)
            
            # TRAINING LOSS
            logger.set_mode('train')
            for b, (xb, yb) in enumerate(train_dl):
                printProgressBar(b, batch_count, prefix='Train loss ', barColor=bcolors.WARNING)
                score_batch(model, loss_func, xb.to(device), yb.to(device), logger=logger, compute_metrics=True)
            printProgressBar(batch_count, batch_count, prefix='Train loss ', barColor=bcolors.WARNING)

        # Save epoch log data to logger history and save model weights
        logger.commit_epoch()
        t.save(model.state_dict(), f'{logger.save_path}/weights/{epoch}')


