import numpy as np


def intersection_over_union(b1, b2):
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
