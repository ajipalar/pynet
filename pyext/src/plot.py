import matplotlib.pyplot as plt
from typing import Any
from .typedefs import Array1d


def marginal(x: Array1d, xlabel: str = "x", ylabel: str = "freq", bins=50, **kwargs):
    x = np.array(x)
    plt.hist(x, bins=bins, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def scatter(x: Array1d, y: Array1d, xlabel: str = "x", ylabel: str = "y", **kwargs):
    plt.scatter(x, y)
    plt.show()
