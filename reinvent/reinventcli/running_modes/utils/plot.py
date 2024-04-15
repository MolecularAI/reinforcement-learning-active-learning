from typing import List, Union
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_histogram(data: List[Union[int, float]], xlabel: str, bins=50, title=None):
    plt.figure()
    plt.hist(x=data, bins=bins, color='#0504aa',
                                alpha=0.7, rwidth=0.95)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)

    return plt.gcf()


def plot_scatter(x: List[Union[int, float]], y: List[Union[int, float]], xlabel: str, ylabel: str, title=None):
    plt.figure()
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    sc = plt.scatter(x, y, c=z, cmap='copper_r', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.colorbar(sc, label="kernel density estimate")

    return plt.gcf()