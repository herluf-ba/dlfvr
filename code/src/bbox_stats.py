import os
import math
import pandas as pd
import numpy as np
from base_model.display import printProgressBar
from torchvision.io import read_image
import matplotlib.pyplot as plt


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


print("Collecting bounding box data")

digit_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
widths = []
heights = []
closest_neighbour_dists = []

img_dir = os.path.join("datasets", 'train')
boxes = pd.read_csv(os.path.join(img_dir, "bbox.csv"))

train_sample_count = 33402
for idx in range(1, train_sample_count):
    image_name = str(idx + 1) + '.png'
    img_path = os.path.join(img_dir, image_name)
    image = read_image(img_path) / 255
    (_, img_height, img_width) = image.shape

    # Get image bounding boxes and normalize coordinates
    mask = boxes["FileName"] == image_name
    labels = [(left / img_width, top / img_height, width / img_width,
               height / img_height)
              for (_, left, top, width,
                   height) in boxes.loc[mask].to_numpy()[:, 1:]]

    digit_distribution[len(labels)] += 1

    for (_, _, width, height) in labels:
        widths.append(width)
        heights.append(height)

    if len(labels) > 1:
        centers = [(left + width / 2.0, top + height / 2.0)
                   for (left, top, width, height) in labels]

        for (i, center) in enumerate(centers):
            closest_neighbour_dist = np.min([
                distance(c, center) if i != j else 100.0
                for (j, c) in enumerate(centers)
            ])
            closest_neighbour_dists.append(closest_neighbour_dist)

    printProgressBar(
        idx,
        train_sample_count,
    )

printProgressBar(
    train_sample_count,
    train_sample_count,
)

# TODO: Compute mean and std deviation of width and height
print(f"Widths\nmean: {np.mean(widths)}\tstd. deviation: {np.std(widths)}")
print(f"Heights\nmean: {np.mean(heights)}\tstd. deviation: {np.std(heights)}")
print(
    f"Closest Neighbour Dist\nmean: {np.mean(closest_neighbour_dists)}\tstd. deviation: {np.std(closest_neighbour_dists)}"
)
print(f"\nDistribution of digit count:\n\t{digit_distribution}")

fig, axs = plt.subplots(2, 2)
axs[0, 0].boxplot(widths)
axs[0, 0].set_title('Width')
axs[0, 1].boxplot(heights)
axs[0, 1].set_title('Height')
axs[1, 0].boxplot(closest_neighbour_dists)
axs[1, 0].set_title('Closest neighbour distance')
axs[1, 1].bar(digit_distribution.keys(), digit_distribution.values())
axs[1, 1].set_title('Box count')
plt.show()