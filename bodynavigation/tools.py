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

import io, os
import json
import copy
import re

import numpy as np
import scipy
import scipy.ndimage
import skimage.transform
import skimage.morphology

import io3d

# dont display some anoying warnings
import warnings

warnings.filterwarnings("ignore", ".* scipy .* output shape of zoom.*")

###########################################
# Crop/Pad/Fraction
###########################################


def getDataPadding(data):
    """
    Returns counts of zeros at the end and start of each axis of N-dim array
    Output for 3D data: [ [pad_start,pad_end], [pad_start,pad_end], [pad_start,pad_end] ]
    """
    ret_l = []
    for dim in range(len(data.shape)):
        widths = []
        s = []
        for dim_s in range(len(data.shape)):
            s.append(slice(0, data.shape[dim_s]))
        for i in range(data.shape[dim]):
            s[dim] = i
            widths.append(np.sum(data[tuple(s)]))
        widths = np.asarray(widths).astype(np.bool)
        pad = [np.argmax(widths), np.argmax(widths[::-1])]  # [pad_before, pad_after]
        ret_l.append(pad)
    return ret_l


def cropArray(
    data, pads, padding_value=0
):  # TODO - skimage.util.crop; convergent programming is funny
    """
    Removes/Adds specified number of values at start and end of every axis of N-dim array

    Input: [ [pad_start,pad_end], [pad_start,pad_end], [pad_start,pad_end] ]
    Positive values crop, Negative values pad.
    """
    pads = [[-p[0], -p[1]] for p in pads]
    return padArray(data, pads, padding_value=padding_value)


def padArray(data, pads, padding_value=0):  # TODO - skimage.util.pad
    """
    Removes/Adds specified number of values at start and end of every axis of N-dim array

    Input: [ [pad_start,pad_end], [pad_start,pad_end], [pad_start,pad_end] ]
    Positive values pad, Negative values crop.
    """
    crops = [[-min(0, p[0]), -min(0, p[1])] for p in pads]
    pads = [[max(0, p[0]), max(0, p[1])] for p in pads]

    # cropping
    s = []
    for dim in range(len(data.shape)):
        s.append(slice(crops[dim][0], data.shape[dim] - crops[dim][1]))
    data = data[tuple(s)]

    # padding
    full_shape = np.asarray(data.shape) + np.asarray(
        [np.sum(pads[dim]) for dim in range(len(pads))]
    )
    out = (np.zeros(full_shape, dtype=data.dtype) + padding_value).astype(data.dtype)
    s = []
    for dim in range(len(data.shape)):
        s.append(slice(pads[dim][0], out.shape[dim] - pads[dim][1]))
    out[tuple(s)] = data

    return out


def getDataFractions(data2d, fraction_defs=[], mask=None, return_slices=False):
    """
    Returns views (in tuple) on 2D array defined by percentages of width and height
    fraction_defs - [{"h":(3/4,1),"w":(0,1)},...]
    mask - used for calculation of width and height based on segmented data
    return_slices - if True returns slice() tuples instead of views into array
    """
    if mask is None:
        height = data2d.shape[0]
        height_offset = 0
        width = data2d.shape[1]
        width_offset = 0
    elif np.sum(mask) == 0:
        height = 0
        height_offset = 0
        width = 0
        width_offset = 0
    else:
        pads = getDataPadding(mask)
        height = data2d.shape[0] - pads[0][1] - pads[0][0]
        height_offset = pads[0][0]
        width = data2d.shape[1] - pads[1][1] - pads[1][0]
        width_offset = pads[1][0]

    def get_index(length, offset, percent):
        return offset + int(np.round(length * percent))

    fractions = []
    slices = []
    for fd in fraction_defs:
        h_s = slice(
            get_index(height, height_offset, fd["h"][0]),
            get_index(height, height_offset, fd["h"][1]) + 1,
        )
        w_s = slice(
            get_index(width, width_offset, fd["w"][0]),
            get_index(width, width_offset, fd["w"][1]) + 1,
        )
        slices.append((h_s, w_s))
        fractions.append(data2d[(h_s, w_s)])

    r = slices if return_slices else fractions
    if len(r) == 1:
        return r[0]
    else:
        return tuple(r)


###########################################
# Resize
###########################################


def resizeScipy(data, toshape, order=1, mode="reflect", cval=0):
    """
    Resize array to shape with scipy.ndimage.zoom

    Use this on big data, because skimage.transform.resize consumes absurd amount of RAM memory
    (many times size of input array), while scipy.ndimage.zoom consumes none.
    scipy.ndimage.zoom also keeps correct dtype of output array.

    Output is a bit (or VERY) wrong, and a lot of minor bugs:
    https://github.com/scipy/scipy/issues/7324
    https://github.com/scipy/scipy/issues?utf8=%E2%9C%93&q=is%3Aopen%20is%3Aissue%20label%3Ascipy.ndimage%20zoom
    """
    order = 0 if (data.dtype == np.bool) else order  # for masks
    zoom = np.asarray(toshape, dtype=np.float) / np.asarray(data.shape, dtype=np.float)
    data = scipy.ndimage.zoom(data, zoom=zoom, order=order, mode=mode, cval=cval)
    if np.any(data.shape != toshape):
        logger.error(
            "Wrong output shape of zoom: %s != %s" % (str(data.shape), str(toshape))
        )
    return data


def resizeSkimage(data, toshape, order=1, mode="reflect", cval=0):
    """
    Resize array to shape with skimage.transform.resize
    Eats memory like crazy (many times size of input array), but very good results.
    """
    dtype = data.dtype  # remember correct dtype

    data = skimage.transform.resize(
        data, toshape, order=order, mode=mode, cval=cval, clip=True, preserve_range=True
    )

    # fix dtype after skimage.transform.resize
    if (data.dtype != dtype) and (dtype in [np.bool, np.integer]):
        data = np.round(data).astype(dtype)
    elif data.dtype != dtype:
        data = data.astype(dtype)

    return data


# TODO - test resize version with RegularGridInterpolator, (only linear and nn order)
# https://scipy.github.io/devdocs/generated/scipy.interpolate.RegularGridInterpolator.html
# https://stackoverflow.com/questions/30056577/correct-usage-of-scipy-interpolate-regulargridinterpolator


def resize(data, toshape, order=1, mode="reflect", cval=0):
    return resizeScipy(data, toshape, order=order, mode=mode, cval=cval)


def resizeWithUpscaleNN(data, toshape, order=1, mode="reflect", cval=0):
    """
    All upscaling is done with 0 order interpolation (Nearest-neighbor) to prevent ghosting effect.
        (Examples of ghosting effect can be seen for example in 3Dircadb1.19)
    Any downscaling is done with given interpolation order.
    If input is binary mask (np.bool) order=0 is forced.
    """
    # calc both resize shapes
    scale = np.asarray(data.shape, dtype=np.float) / np.asarray(toshape, dtype=np.float)
    downscale_shape = np.asarray(toshape, dtype=np.int).copy()
    if scale[0] > 1.0:
        downscale_shape[0] = data.shape[0]
    if scale[1] > 1.0:
        downscale_shape[1] = data.shape[1]
    if scale[2] > 1.0:
        downscale_shape[2] = data.shape[2]
    upscale_shape = np.asarray(toshape, dtype=np.int).copy()

    # downscale with given interpolation order
    data = resize(data, downscale_shape, order=order, mode=mode, cval=cval)

    # upscale with 0 order interpolation
    if not np.all(downscale_shape == upscale_shape):
        data = resize(data, upscale_shape, order=0, mode=mode, cval=cval)

    return data


###########################################
# Segmentation
###########################################


def getSphericalMask(size=5, spacing=[1, 1, 1]):
    """ Size is in mm """
    shape = (
        np.asarray([size] * 3, dtype=np.float) / np.asarray(spacing, dtype=np.float)
    ).astype(np.int)
    shape[shape < 1] = 1
    mask = skimage.morphology.ball(51, dtype=np.float)
    mask = resizeSkimage(mask, shape, order=1, mode="edge", cval=0) > 0.001
    return mask


def getDiskMask(size=5, spacing=[1, 1, 1]):
    """ Size is in mm """
    shape = (
        np.asarray([size] * 3, dtype=np.float) / np.asarray(spacing, dtype=np.float)
    ).astype(np.int)
    shape[shape < 1] = 1
    shape[0] = 1
    mask = np.expand_dims(skimage.morphology.disk(51, dtype=np.bool), axis=0)
    mask = resizeSkimage(mask, shape, order=1, mode="edge", cval=0) > 0.001
    return mask


def binaryClosing(data, structure, cval=0):
    """
    Does scipy.ndimage.morphology.binary_closing() without losing data near borders
    Big sized structures can make this take a long time
    """
    padding = np.max(structure.shape)
    tmp = (
        np.zeros(np.asarray(data.shape) + padding * 2, dtype=data.dtype) + cval
    ).astype(np.bool)
    tmp[padding:-padding, padding:-padding, padding:-padding] = data
    tmp = scipy.ndimage.morphology.binary_closing(tmp, structure=structure)
    return tmp[padding:-padding, padding:-padding, padding:-padding]


def binaryFillHoles(data, z_axis=False, y_axis=False, x_axis=False):
    """
    Does scipy.ndimage.morphology.binary_fill_holes() as if at the start and end of [z/y/x]-axis is solid wall
    """

    if not (z_axis or x_axis or y_axis):
        return scipy.ndimage.morphology.binary_fill_holes(data)

    # fill holes on z-axis
    if z_axis:
        tmp = np.ones((data.shape[0] + 2, data.shape[1], data.shape[2]))
        tmp[1:-1, :, :] = data
        tmp = scipy.ndimage.morphology.binary_fill_holes(tmp)
        data = tmp[1:-1, :, :]

    # fill holes on y-axis
    if y_axis:
        tmp = np.ones((data.shape[0], data.shape[1] + 2, data.shape[2]))
        tmp[:, 1:-1, :] = data
        tmp = scipy.ndimage.morphology.binary_fill_holes(tmp)
        data = tmp[:, 1:-1, :]

    # fill holes on x-axis
    if x_axis:
        tmp = np.ones((data.shape[0], data.shape[1], data.shape[2] + 2))
        tmp[:, :, 1:-1] = data
        tmp = scipy.ndimage.morphology.binary_fill_holes(tmp)
        data = tmp[:, :, 1:-1]

    return data


def regionGrowing(data3d, seeds, mask, spacing=None, max_dist=-1, mode="watershed"):
    """
    Does not ignore 'geography' of data when calculating 'distances' growing regions.
    Has 2 modes, 'random_walker' and 'watershed'.

    data3d - data3d or binary mask to be segmented
    seeds - seeds, are converted to np.int8
    mask - extremely important for 'random_walker', accidentally processing whole data eats 10s of GB.
    spacing - voxel spacing, if None cube spacing is assumed.
    max_dist - tries to limit maximal growth distance from seeds (ignores 'geography of data')
    mode - 'random_walker'/'watershed'

    'random_walker' mode is based on diffusion of probability.
    Should not ignore brightness of pixels (I think?) - different brightness == harder diffusion
    A lot more memory required then 'watershed'. (1.7GB vs 4.2GB MAXMEM used)

    'watershed' mode based on filling hypothetical basins in data with liquid.
    In problem of segmentation in CT data, is only useful in very specific situations.
    (grayscale data3d doesnt work in very useful way with this).
    If used together with spacing parameter, a lot more memory is required (1.7GB vs 4.3GB MAXMEM used).

    Lowest possible used memory is when mode='watershed' and spacing=None
    """
    # note - large areas that are covered by seeds do not increase memory requirements
    #        (works almost as if they had mask == 0)
    seeds = seeds.astype(np.int8).copy()
    mask = mask.copy()

    # limit max segmentation distance
    if max_dist > 0:
        mask[
            scipy.ndimage.morphology.distance_transform_edt(
                seeds == 0, sampling=spacing
            )
            > max_dist
        ] = 0

    # remove sections in mask that are not connected to any seeds # TODO - test if this lowers memory requirements
    mask = skimage.measure.label(mask, background=0)
    tmp = mask.copy()
    tmp[seeds == 0] = 0
    for l in np.unique(tmp)[1:]:
        mask[mask == l] = -1
    mask = mask == -1
    del tmp

    # if only one seed, return everything connected to it (done in last step).
    unique = np.unique(seeds)[1:]
    if len(unique) == 1:
        return mask.astype(np.int8) * unique[0]

    # segmentation
    if mode not in ["random_walker", "watershed"]:
        logger.warning(
            "Invalid region growing mode '%s', defaulting to 'random_walker'"
            % str(mode)
        )
        mode = "random_walker"

    if mode == "random_walker":
        seeds[mask == 0] = -1
        seeds = skimage.segmentation.random_walker(
            data3d, seeds, mode="cg_mg", copy=False, spacing=spacing
        )
        seeds[seeds == -1] = 0

    elif (
        mode == "watershed"
    ):  # TODO - maybe more useful if edge filter is done first, when using grayscale data??
        # resize data to cube spacing
        if spacing is not None:
            shape_orig = data3d.shape
            shape_cube = np.asarray(data3d.shape, dtype=np.float) * np.asarray(
                spacing, dtype=np.float
            )  # 1x1x1mm
            shape_cube = (shape_cube / np.min(spacing)).astype(
                np.int
            )  # upscale target size, so there is no loss in quality

            order = 0 if (data3d.dtype == np.bool) else 1  # for masks
            data3d = resize(data3d, shape_cube, order=order, mode="reflect")
            mask = resize(mask, shape_cube, order=0, mode="reflect")
            tmp = seeds
            seeds = np.zeros(shape_cube, dtype=seeds.dtype)
            for s in np.unique(tmp)[1:]:
                seeds[resize(tmp == s, shape_cube, order=0, mode="reflect")] = s
            del tmp

        seeds = skimage.morphology.watershed(data3d, seeds, mask=mask)

        # resize back to original spacing/shape
        if spacing is not None:
            tmp = seeds
            seeds = np.zeros(shape_orig, dtype=seeds.dtype)
            for s in np.unique(tmp)[1:]:
                seeds[resize(tmp == s, shape_orig, order=0, mode="reflect")] = s

    return seeds


###################
# Memory saving
###################


def compressArray(mask):
    """ Compresses numpy array from RAM to RAM """
    mask_comp = io.BytesIO()
    np.savez_compressed(mask_comp, mask)
    return mask_comp


def decompressArray(mask_comp):
    """ Decompresses numpy array from RAM to RAM """
    mask_comp.seek(0)
    return np.load(mask_comp)["arr_0"]


def toMemMap(data3d, filepath):
    """
    Move numpy array from RAM to file
    np.memmap might not work with some functions that np.array would have worked with. Sometimes
    can even crash without error.
    """
    data3d_tmp = data3d
    data3d = np.memmap(filepath, dtype=data3d.dtype, mode="w+", shape=data3d.shape)
    data3d[:] = data3d_tmp[:]
    del data3d_tmp
    data3d.flush()
    return data3d


def delMemMap(data3d):
    """ Deletes file used for memmap. Trying to use array after this runs will crash Python """
    filename = copy.deepcopy(data3d.filename)
    data3d.flush()
    data3d._mmap.close()
    del data3d
    os.remove(filename)


def concatenateMemMap(A, B):
    """
    concatenate memmap along axis=0
    A - must be np.memmap
    B - must be same dtype as A, but does not need to be memmap
    """
    old_shape = tuple(A.shape)
    new_shape = tuple([A.shape[0] + B.shape[0]] + list(A.shape)[1:])
    A = np.memmap(A.filename, dtype=A.dtype, mode="r+", shape=new_shape, order="C")
    A[old_shape[0] :, :] = B
    return A


###################
# Misc
###################


def polyfit3D(points, dtype=np.int, deg=3):
    z, y, x = zip(*points)
    z_new = list(range(z[0], z[-1] + 1))

    zz1 = np.polyfit(z, y, deg)
    f1 = np.poly1d(zz1)
    y_new = f1(z_new)

    zz2 = np.polyfit(z, x, deg)
    f2 = np.poly1d(zz2)
    x_new = f2(z_new)

    points = [
        tuple(np.asarray([z_new[i], y_new[i], x_new[i]]).astype(dtype))
        for i in range(len(z_new))
    ]
    return points


class NumpyEncoder(json.JSONEncoder):
    """
    Fixes saving numpy arrays into json

    Example:
    a = np.array([1, 2, 3])
    print(json.dumps({'aa': [2, (2, 3, 4), a], 'bb': [2]}, cls=NumpyEncoder))
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def firstNonzero(data3d, axis, invalid_val=-1):
    """
    Returns (N-1)D array with indexes of first non-zero elements along defined axis
    """
    mask = data3d != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def flip(
    m, axis
):  # TODO - replace with np.flip when there are no dependency problems with numpy>=1.12
    """
    Copy of numpy.flip, which was added in numpy 1.12.0
    (Had to create copy of this, because I got problems with dependencies)
    """
    if not hasattr(m, "ndim"):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError(
            "axis=%i is invalid for the %i-dimensional input array" % (axis, m.ndim)
        )
    return m[tuple(indexer)]


def useDatasetMod(data3d, misc):
    """
    This function should be used to fix loading of files from datasets with wierd format.
    Example: sliver07 dataset has fliped z-axis
    """
    # def missing misc variables
    misc["flip_z"] = False if ("flip_z" not in misc) else misc["flip_z"]

    # do misc
    if misc["flip_z"]:
        data3d = flip(data3d, axis=0)

    return data3d


def readCompoundMask(path_list):
    # load masks
    mask, mask_metadata = io3d.datareader.read(path_list[0], dataplus_format=False)
    mask = mask > 0  # to np.bool
    for p in path_list[1:]:
        tmp, _ = io3d.datareader.read(p, dataplus_format=False)
        tmp = tmp > 0  # to np.bool
        mask[tmp] = 1
    return mask, mask_metadata


def naturalSort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def getBiggestObjects(mask, N=1):
    if np.sum(mask) == 0:
        return mask

    # get labels and counts of objects
    mask_label = skimage.measure.label(mask, background=0)
    unique, counts = np.unique(mask_label, return_counts=True)
    unique = list(unique[1:])
    counts = list(counts[1:])

    # get labels of biggest objects
    biggest_labels = []
    for n in range(N):
        biggest_idx = list(counts).index(max(counts))
        biggest_labels.append(unique[biggest_idx])
        del (unique[biggest_idx], counts[biggest_idx])

    # return only biggest N objects
    mask[:] = 0
    for l in biggest_labels:
        mask[mask_label == l] = 1
    return mask
