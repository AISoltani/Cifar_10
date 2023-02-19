# Author: Hamidreza Nademi

import numpy as np
import matplotlib.pyplot as plt


def plot_data(inputs, labels, sample_icon, sample_color, title):
    """ Plot dataset """
    x0, x1 = zip(*inputs)
    uniqe_lbls, dict_data = set(labels), {}
    plt.figure()

    for label in uniqe_lbls:
        dict_data[label] = []

    for i in range(len(labels)):
        dict_data[labels[i]].append(inputs[i])

    for label, color in zip(dict_data.keys(), sample_color):
        x0, x1 = zip(*dict_data[label])
        plt.plot(x0, x1, sample_icon, color=color, label=f'class {int(label)}')

    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(title)
    plt.legend()
