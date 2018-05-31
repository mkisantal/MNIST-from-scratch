from matplotlib import pyplot as plt
import numpy as np


def show(flattened, ax=None):
    sample = np.reshape(flattened, [28, 28])
    plt.ion()
    if ax is not None:
        ax.imshow(sample)
    else:
        plt.imshow(sample)
    # plt.show()


def barplot(vector, ax=None):
    indices = range(vector.size)

    if len(vector.shape) != 1:
        vector = vector.tolist()[0]
    else:
        vector = vector.tolist()
    if ax is not None:
        ax.bar(indices, vector)
    else:
        plt.bar(indices, vector)
    return vector


def accuracy(inputs, targets, net):
    correctly_classified = 0
    for i in range(inputs.shape[0]):
        output = net.forward(inputs[i, :])
        if np.argmax(output) == targets[i]:
            correctly_classified += 1
    return correctly_classified / float(inputs.shape[0])


def misclassified(inputs, targets, net):
    missed_samples = []
    for i in range(inputs.shape[0]):
        output = net.forward(inputs[i, :])
        if np.argmax(output) != targets[i]:
            missed_samples.append(i)
    return missed_samples

def inference(sample, inputs, targets, net):
    input = inputs[sample, :]
    output = net.forward(input)

    # plot input image and output distribution
    target = targets[sample]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    show(input, ax1)
    barplot(output, ax2)
    ax1.set_title('Input: {}'.format(target))
    ax2.set_title('Output')


def show_hidden(i, net):

    # plot input image and output distribution
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    show(net.Wih[:, i], ax1)
    barplot(net.Who[i, :], ax2)
    ax1.set_title('Input to hidden weights.')
    ax2.set_title('Hidden to output weights.')


