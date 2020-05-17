import numpy as np


def read_imsil(fname):
    """
    Read data from imsil file.

    :param fname: path of the file.
    :return x, y, z: 2D arrays of x, y, and the function value.
    """
    array = np.loadtxt(fname)
    x = array[1:,0]
    y = array[0,1:]
    z = array[1:,1:]

    x, y = np.meshgrid(x, y)

    return x, y, z