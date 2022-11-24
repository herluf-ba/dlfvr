import numpy as np
import torch as t
from svhn import batch_extract_bounding_box


def intersection_over_union(predictions, labels):
    prediction_bbs = t.clamp(batch_extract_bounding_box(predictions), min=0.0)
    label_bbs = t.clamp(batch_extract_bounding_box(labels), min=0.0)

    print(f'{prediction_bbs=}\n{label_bbs=}')
    ## (64, 4, 4)
    ## (bs, S*S, coords)

    prediction_areas = t.multiply(prediction_bbs[:, :, 2], prediction_bbs[:, :,
                                                                          3])
    label_areas = t.multiply(label_bbs[:, :, 2], label_bbs[:, :, 3])

    intersections = t.cat(
        (
            ## largest left coordinates
            t.maximum(prediction_bbs[:, :, 0], label_bbs[:, :, 0]),
            ## largest top coordinates
            t.maximum(prediction_bbs[:, :, 1], label_bbs[:, :, 1]),
            ## smallest right coordinates
            t.minimum(
                t.multiply(prediction_bbs[:, :, 0], prediction_bbs[:, :, 2]),
                t.multiply(label_bbs[:, :, 0], label_bbs[:, :, 2])),
            ## smallest bottom coordinates
            t.minimum(
                t.multiply(prediction_bbs[:, :, 1], prediction_bbs[:, :, 3]),
                t.multiply(label_bbs[:, :, 1], label_bbs[:, :, 3]))),
        axis=0)

    print(f'{intersections=}')
    intersection_areas = []

    # Compute intersection bounding box
    intersection = [
        max(b1[0], b2[0]),
        max(b1[1], b2[1]),
        min(b1[2], b2[2]),
        min(b1[3], b2[3]),
    ]

    # Compute areas of b1, b2 and instersection
    intersection_area = max(0, intersection[2] + intersection[0] + 1) * max(
        0, intersection[3] - intersection[1] + 1)

    b1_area = max(0, b1[2] + b1[0] + 1) * max(0, b1[3] - b1[1] + 1)
    b2_area = max(0, b2[2] + b2[0] + 1) * max(0, b2[3] - b2[1] + 1)

    # Compute intersection over union
    return intersection_area / float(b1_area + b2_area - intersection_area)
