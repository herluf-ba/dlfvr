import numpy as np
import torch as t
from settings import S
from torchvision.ops import complete_box_iou_loss
from svhn import batch_extract_bounding_box, batch_extract_confidence

BOXES_PER_PRED = S * S
LEFT = 0
TOP = 1
WIDTH = 2
HEIGHT = 3
RIGHT = 2
BOTTOM = 3


def intersection_over_union(predictions, labels):
    ## Create a mask to remove all boxes where label confidence is zero
    mask = batch_extract_confidence(labels).reshape(-1, 1).expand(
        -1, BOXES_PER_PRED)

    ## Untrained models can have negative widths and heights so clamp to zero
    label_bbs = t.clamp(batch_extract_bounding_box(labels),
                        min=0.0).reshape(-1, BOXES_PER_PRED)
    prediction_bbs = t.clamp(batch_extract_bounding_box(predictions),
                             min=0.0).reshape(-1, BOXES_PER_PRED)

    ## Filter off all the bounding boxes where the label is zero (no labeled digit)
    prediction_bbs = prediction_bbs[mask == 1.0].reshape(-1, BOXES_PER_PRED)
    label_bbs = label_bbs[mask == 1.0].reshape(-1, BOXES_PER_PRED)
    label_bbs = label_bbs.reshape(-1, BOXES_PER_PRED)

    ## Compute rigth and bottom coordinates rather than width and height
    prediction_bbs[:, RIGHT] = t.add(prediction_bbs[:, LEFT],
                                     prediction_bbs[:, WIDTH])
    prediction_bbs[:, BOTTOM] = t.add(prediction_bbs[:, TOP],
                                      prediction_bbs[:, HEIGHT])
    label_bbs[:, RIGHT] = t.add(label_bbs[:, LEFT], label_bbs[:, WIDTH])
    label_bbs[:, BOTTOM] = t.add(label_bbs[:, TOP], label_bbs[:, HEIGHT])

    ## Compute mean of iou loss
    iou_loss = complete_box_iou_loss(prediction_bbs,
                                     label_bbs,
                                     reduction='mean')
    return iou_loss
