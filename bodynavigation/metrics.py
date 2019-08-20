#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
from __future__ import print_function  # print("text")
from __future__ import division  # 2/3 == 0.666; 2//3 == 0
from __future__ import (
    absolute_import,
)  # 'import submodule2' turns into 'from . import submodule2'
from builtins import range  # replaces range with xrange

import logging

logger = logging.getLogger(__name__)

import numpy as np
import scipy.ndimage

from .tools import getDataPadding, cropArray

"""
Big part of this file is refractored lisa.volumetry_evaluation
"""


def voe(vol1, vol2):
    """ VOE[0;1] - Volume Overlap Error """
    df = vol1 != vol2
    intersection = float(np.sum(df))
    union = float(np.sum(vol1) + np.sum(vol2))
    return intersection / union


def vd(vol1, vol2):
    """ VD[-1;1] - Volume Difference """
    return float(np.sum(vol2) - np.sum(vol1)) / float(np.sum(vol1))


def dice(vol1, vol2):
    """ Dice[0;1] - Dice coefficient. Dice = 1.0 - VOE """
    a = np.sum(vol1[vol2])
    b = np.sum(vol1)
    c = np.sum(vol2)
    return (2 * a) / (b + c)


def _get_border(vol):
    tmp = np.ones(
        (vol.shape[0] + 2, vol.shape[1] + 2, vol.shape[2] + 2), dtype=vol.dtype
    )
    tmp[1:-1, 1:-1, 1:-1] = vol

    erode = scipy.ndimage.binary_erosion(tmp, np.ones((3, 3, 3)))
    border = tmp ^ erode

    border = border[1:-1, 1:-1, 1:-1]
    return border


def distanceMetrics(vol1, vol2, voxelsize_mm):
    """
    avgd[mm] - Average symmetric surface distance
    rmsd[mm] - RMS symmetric surface distance
    maxd[mm] - Maximum symmetric surface distance
    """
    # crop data to reduce computation time
    pads1 = getDataPadding(vol1)
    pads2 = getDataPadding(vol2)
    pads = [
        [min(pads1[0][0], pads2[0][0]), min(pads1[0][1], pads2[0][1])],
        [min(pads1[1][0], pads2[1][0]), min(pads1[1][1], pads2[1][1])],
        [min(pads1[2][0], pads2[2][0]), min(pads1[2][1], pads2[2][1])],
    ]
    vol1 = cropArray(vol1, pads)
    vol2 = cropArray(vol2, pads)

    # compute borders and distances
    border1 = _get_border(vol1)
    border2 = _get_border(vol2)
    # pyed = sed3.sed3(vol1, seeds=border1); pyed.show()
    b1dst = scipy.ndimage.morphology.distance_transform_edt(
        border1 == 0, sampling=voxelsize_mm
    )
    b2dst = scipy.ndimage.morphology.distance_transform_edt(
        border2 == 0, sampling=voxelsize_mm
    )
    dst_b1_to_b2 = border2 * b1dst
    dst_b2_to_b1 = border1 * b2dst
    dst_12 = dst_b1_to_b2[border2]
    dst_21 = dst_b2_to_b1[border1]
    dst_both = np.append(dst_12, dst_21)

    # compute metrics
    avgd = np.average(dst_both)
    rmsd = np.average(dst_both ** 2)
    maxd = max(np.max(dst_b1_to_b2), np.max(dst_b2_to_b1))

    return avgd, rmsd, maxd


def compareVolumes(vol1, vol2, voxelsize_mm=np.asarray([1, 1, 1])):
    """
    computes metrics
    vol1: reference
    vol2: segmentation
    """
    # convert to np.bool
    vol1 = vol1 > 0
    vol2 = vol2 > 0

    # compute metrics
    evaluation = {}
    evaluation["vd"] = vd(vol1, vol2)
    evaluation["voe"] = voe(vol1, vol2)
    evaluation["dice"] = dice(vol1, vol2)
    avgd, rmsd, maxd = distanceMetrics(vol1, vol2, voxelsize_mm)
    evaluation["avgd"] = avgd
    evaluation["rmsd"] = rmsd
    evaluation["maxd"] = maxd

    return evaluation
