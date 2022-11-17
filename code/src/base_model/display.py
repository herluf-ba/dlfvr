import matplotlib.pyplot as plt
import matplotlib.patches as patches


def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=100,
                     fill='━',
                     printEnd="\r"):
    '''Print iterations progress'''
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def plot_img_vanilla(image, labels):
    '''Plots a untransformed image with labels'''
    fig, ax = plt.subplots()
    ax.imshow(image.numpy().transpose([1, 2, 0]), cmap='gray')

    for (_, left, top, width, height) in labels:
        ax.add_patch(
            patches.Rectangle((left, top),
                              width,
                              height,
                              linewidth=1,
                              edgecolor='r',
                              facecolor='none'))
    plt.show()


def plot_img(image, labels):
    '''Plots a transformed image with labels'''
    fig, ax = plt.subplots()
    ax.imshow(image.numpy().transpose([1, 2, 0]), cmap='gray')

    for x in range(labels.shape[1]):
        for y in range(labels.shape[2]):
            top = labels[1][x][y]
            left = labels[2][x][y]
            width = labels[4][x][y]
            height = labels[3][x][y]
            ax.add_patch(
                patches.Rectangle((left, top),
                                  width,
                                  height,
                                  linewidth=1,
                                  edgecolor='r',
                                  facecolor='none'))
    plt.show()