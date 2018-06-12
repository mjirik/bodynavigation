#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
from __future__ import print_function   # print("text")
from __future__ import division         # 2/3 == 0.666; 2//3 == 0
from __future__ import absolute_import  # 'import submodule2' turns into 'from . import submodule2'
from builtins import range              # replaces range with xrange

import logging
logger = logging.getLogger(__name__)

import io
import json

import numpy as np
import scipy
import scipy.ndimage
import skimage.transform
import skimage.morphology

def getSphericalMask(shape=[3,3,3], spacing=[1,1,1]):
    shape = (np.asarray(shape, dtype=np.float)/np.asarray(spacing, dtype=np.float)).astype(np.int)
    shape[0] = max(shape[0], 1); shape[1] = max(shape[1], 1); shape[2] = max(shape[2], 1)
    mask = skimage.morphology.ball(21, dtype=np.bool)
    mask = skimage.transform.resize(
        mask, np.asarray(shape).astype(np.int), order=1,
        mode="constant", cval=0, clip=True, preserve_range=True
        ).astype(np.bool)
    return mask

def binaryClosing(data, structure, cval=0):
    """
    Does scipy.ndimage.morphology.binary_closing() without losing data near borders
    Big sized structures can make this take a long time
    """
    padding = np.max(structure.shape)
    tmp = np.zeros(np.asarray(data.shape)+padding*2) + cval
    tmp[padding:-padding,padding:-padding,padding:-padding] = data
    tmp = scipy.ndimage.morphology.binary_closing(tmp, structure=structure)
    return tmp[padding:-padding,padding:-padding,padding:-padding]

def binaryFillHoles(data, z_axis=False, y_axis=False, x_axis=False):
    """
    Does scipy.ndimage.morphology.binary_fill_holes() as if at the start and end of [z/y/x]-axis is solid wall
    """

    if not (z_axis or x_axis or y_axis):
        return scipy.ndimage.morphology.binary_fill_holes(data)

    # fill holes on z-axis
    if z_axis:
        tmp = np.ones((data.shape[0]+2, data.shape[1], data.shape[2]))
        tmp[1:-1,:,:] = data;
        tmp = scipy.ndimage.morphology.binary_fill_holes(tmp)
        data = tmp[1:-1,:,:]

    # fill holes on y-axis
    if y_axis:
        tmp = np.ones((data.shape[0], data.shape[1]+2, data.shape[2]))
        tmp[:,1:-1,:] = data;
        tmp = scipy.ndimage.morphology.binary_fill_holes(tmp)
        data = tmp[:,1:-1,:]

    # fill holes on x-axis
    if x_axis:
        tmp = np.ones((data.shape[0], data.shape[1], data.shape[2]+2))
        tmp[:,:,1:-1] = data;
        tmp = scipy.ndimage.morphology.binary_fill_holes(tmp)
        data = tmp[:,:,1:-1]

    return data

def compressArray(mask):
    logger.debug("compressArray()")
    mask_comp = io.BytesIO()
    np.savez_compressed(mask_comp, mask)
    return mask_comp

def decompressArray(mask_comp):
    logger.debug("decompressArray()")
    mask_comp.seek(0)
    return np.load(mask_comp)['arr_0']

def getDataPadding(data):
    """
    Returns counts of zeros at the end and start of each axis of N-dim array
    Output for 3D data: [ [pad_start,pad_end], [pad_start,pad_end], [pad_start,pad_end] ]
    """
    ret_l = []
    for dim in range(len(data.shape)):
        widths = []; s = []
        for dim_s in range(len(data.shape)):
            s.append(slice(0,data.shape[dim_s]))
        for i in range(data.shape[dim]):
            s[dim] = i; widths.append(np.sum(data[tuple(s)]))
        widths = np.asarray(widths).astype(np.bool)
        pad = [np.argmax(widths), np.argmax(widths[::-1])] # [pad_before, pad_after]
        ret_l.append(pad)
    return tuple(ret_l)

def cropArray(data, pads):
    """
    Removes specified number of values at start and end of every axis from N-dim array
    Pads for 3D data: [ [pad_start,pad_end], [pad_start,pad_end], [pad_start,pad_end] ]
    Does not create copy of input, but creates view on section of input!
    """
    s = []
    for dim in range(len(data.shape)):
        s.append( slice(pads[dim][0],data.shape[dim]-pads[dim][1]) )
    return data[tuple(s)]

def padArray(data, pads, padding_value=0):
    """
    Pads N-dim array with specified value
    Pads for 3D data: [ [pad_start,pad_end], [pad_start,pad_end], [pad_start,pad_end] ]
    """
    full_shape = np.asarray(data.shape) + np.asarray([ np.sum(pads[dim]) for dim in range(len(pads))])
    out = np.zeros(full_shape, dtype=data.dtype) + padding_value
    s = []
    for dim in range(len(data.shape)):
        s.append( slice( pads[dim][0], out.shape[dim]-pads[dim][1] ) )
    out[tuple(s)] = data
    return out

def polyfit3D(points, dtype=np.int, deg=3):
    z, y, x = zip(*points)
    z_new = list(range(z[0], z[-1]+1))

    zz1 = np.polyfit(z, y, deg)
    f1 = np.poly1d(zz1)
    y_new = f1(z_new)

    zz2 = np.polyfit(z, x, deg)
    f2 = np.poly1d(zz2)
    x_new = f2(z_new)

    points = [ tuple(np.asarray([z_new[i], y_new[i], x_new[i]]).astype(dtype)) for i in range(len(z_new)) ]
    return points

def growRegion(region, mask, iterations=1):
    region[ mask == 0 ] = 0

    kernel1 = np.zeros((3,3,3), dtype=np.bool)
    kernel1[:,1,1] = 1; kernel1[1,1,:] = 1; kernel1[1,:,1] = 1
    kernel2 = np.zeros((3,3,3), dtype=np.bool)
    kernel2[:,1,1] = 1; kernel2[1,:,:] = 1
    for i in range(iterations):
        if np.sum(region) == 0: break
        kernel = kernel1 if i%2 == 0 else kernel2
        region = scipy.ndimage.binary_dilation(region, structure=kernel, mask=mask)

    return region

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
