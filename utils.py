from matplotlib import pyplot as plt
import numpy as np


def show(flattened):
    sample = np.reshape(flattened, [28, 28])
    plt.ion()
    plt.imshow(sample)
    plt.show()
