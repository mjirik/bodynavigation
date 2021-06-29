import os.path as op
import sys

sys.path.append(op.expanduser("~/projects/bodynavigation"))

import matplotlib.pyplot as plt
import numpy as np


import csv
import os
from . import lines
import copy
import glob
import io3d
import pandas as pd
import scipy
import scipy.signal
import sklearn
import sklearn.naive_bayes
import sklearn.tree
import sklearn.mixture

# tohle je tu ze zoufalství, správně by se mělo používat bodynavigation.bo
try:
    import bodynavigation.body_navigation
    from importlib import reload

    reload(bodynavigation.body_navigation)

except:
    pass
import imtools
import sed3
from imtools import qmisc, misc, ml


def read_scan(dataset, scannum):
    """Load selected scan from selected dataset.

    Args:
        dataset (string): "3Dircadb1" or "sliver07"
        scannum (int): Number of the scan this loads

    Returns:
        ss: bodynavigation data
        data3d_orig: orig data
    """
    data3d_orig = io3d.read_dataset(dataset, "data3d", scannum)
    ss = bodynavigation.body_navigation.BodyNavigation(
        data3d_orig["data3d"], data3d_orig["voxelsize_mm"]
    )
    voxelsize = data3d_orig["voxelsize_mm"]
    return ss, data3d_orig["data3d"], voxelsize


def visualize(seg, data3d_orig):
    """Show segmentation on original data using sed3 visualizer.

    Args:
        seg: data after segmentation
        data3d_orig: orig data
    """
    sed3.show_slices(data3d_orig["data3d"], seg, slice_step=10, axis=0)
    plt.show()
