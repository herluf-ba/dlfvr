import matplotlib.pyplot as plt
import matplotlib.patches as patches


class bcolors:
    HEADER = '\033[95m'
    BG = '\033[30m'
    OKBLUE = '\033[94m'
    PURPLE = '\033[35m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=50,
                     fill='━',
                     printEnd="\r"):
    '''Print iterations progress'''
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = f"{bcolors.PURPLE}{fill * filledLength}{bcolors.BG}{'━' * (length - filledLength)}{bcolors.ENDC}"
    print(f'\r {prefix} {bar} {percent}% {suffix}', end=printEnd)
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