#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
from __future__ import print_function   # print("text")
from __future__ import division         # 2/3 == 0.666; 2//3 == 0
from __future__ import absolute_import  # 'import submodule2' turns into 'from . import submodule2'
from builtins import range              # replaces range with xrange

import logging
logger = logging.getLogger(__name__)
import traceback

import sys, os
import io
import copy
from operator import itemgetter
from itertools import groupby
import json

import numpy as np
import scipy
import scipy.ndimage
import skimage.measure
import skimage.transform
import skimage.morphology
import skimage.segmentation
import skimage.feature

import io3d
import sed3

#
# Utility Functions
#

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

#
# Main class
#

class OrganDetection(object):
    """
    * getORGAN()
        - for when you create class object (standard usage)
        - resizes output to corect shape (unless you use "raw=True")
        - saves (compressed) output for future calls
    * _getORGAN()
        - for using algorithm directly without class object: OrganDetection._getORGAN()
        - does not resize output
        - does not save output for future calls
        - all required data is in function parameters
    * internal voxelsize is self.spacing and is first normalized and then data is scaled to [normed-z, 1, 1]
    """

    NORMED_FATLESS_BODY_SIZE = [200,300] # normalized size of fatless body on [Y,X] in mm
    # [186.9, 247.4]mm - 3Dircadb1.1
    # [180.6, 256.4]mm - 3Dircadb1.2
    # [139.2, 253.1]mm - 3Dircadb1.19
    # [157.2, 298.8]mm - imaging.nci.nih.gov_NSCLC-Radiogenomics-Demo_1.3.6.1.4.1.14519.5.2.1.4334.1501.332134560740628899985464129848
    # [192.2, 321.4]mm - imaging.nci.nih.gov_NSCLC-Radiogenomics-Demo_1.3.6.1.4.1.14519.5.2.1.4334.1501.117528645891554472837507616577
    # [153.3, 276.3]mm - imaging.nci.nih.gov_NSCLC-Radiogenomics-Demo_1.3.6.1.4.1.14519.5.2.1.4334.1501.164951610473301901732855875499
    # [190.4, 304.6]mm - imaging.nci.nih.gov_NSCLC-Radiogenomics-Demo_1.3.6.1.4.1.14519.5.2.1.4334.1501.252450948398764180723210762820

    def __init__(self, data3d=None, voxelsize=[1,1,1], size_normalization=True, rescale=True):
        """
        * Values of input data should be in HU units (or relatively close). [air -1000, water 0]
            https://en.wikipedia.org/wiki/Hounsfield_scale
        * All coordinates and sizes are in [Z,Y,X] format
        * Expecting data3d to be corectly oriented
        * Voxel size is in mm
        """

        # empty undefined values
        self.data3d = None
        self.spacing = np.asarray([1,1,1], dtype=np.float)
        self.cut_shape = (0,0,0)
        self.padding = [[0,0],[0,0],[0,0]]
        self.orig_voxelsize = np.asarray([1,1,1], dtype=np.float)
        self.orig_coord_scale = np.asarray([1,1,1], dtype=np.float)
        self.orig_coord_intercept = np.asarray([0,0,0], dtype=np.float)

        # compressed masks - example: compression lowered memory usage to 0.042% for bones
        self.masks_comp = {
            "body":None,
            "fatlessbody":None,
            "lungs":None,
            "diaphragm":None,
            "bones":None,
            "abdomen":None,
            "vessels":None,
            "aorta":None,
            "venacava":None,
            }

        # statistics and models
        self.stats = {
            "bones":None,
            "vessels":None
            }

        # init with data3d
        if data3d is not None:
            self.data3d, self.spacing, self.cut_shape, self.padding = self.prepareData( \
                data3d, voxelsize, size_normalization=size_normalization, rescale=rescale)

            # for recalculating coordinates to output format ( vec*scale + intercept )
            self.orig_voxelsize = np.asarray(voxelsize, dtype=np.float)
            self.orig_coord_scale = np.asarray([ \
                self.cut_shape[0]/self.data3d.shape[0], \
                self.cut_shape[1]/self.data3d.shape[1], \
                self.cut_shape[2]/self.data3d.shape[2] ], dtype=np.float) # [z,y,x] - scale coords of cut and resized data
            self.orig_coord_intercept = np.asarray([ \
                self.padding[0][0], \
                self.padding[1][0], \
                self.padding[2][0] ], dtype=np.float) # [z,y,x] - move coords of just cut data

    @classmethod
    def fromReadyData(cls, data3d, data3d_info, masks={}, stats={}):
        """ For super fast testing """
        obj = cls()

        obj.data3d = data3d
        obj.spacing = np.asarray(data3d_info["spacing"], dtype=np.float)
        obj.cut_shape = np.asarray(data3d_info["cut_shape"], dtype=np.float)
        obj.padding = copy.deepcopy(data3d_info["padding"])
        obj.orig_voxelsize = np.asarray(data3d_info["orig_voxelsize"], dtype=np.float)
        obj.orig_coord_scale = np.asarray(data3d_info["orig_coord_scale"], dtype=np.float)
        obj.orig_coord_intercept = np.asarray(data3d_info["orig_coord_intercept"], dtype=np.float)

        for part in masks:
            if part not in obj.masks_comp:
                logger.warning("'%s' is not valid mask name!" % part)
                continue
            obj.masks_comp[part] = masks[part]

        for part in stats:
            if part not in obj.stats:
                logger.warning("'%s' is not valid part stats name!" % part)
                continue
            obj.stats[part] = copy.deepcopy(stats[part])

        return obj

    @classmethod
    def fromDirectory(cls, path):
        logger.info("Loading already processed data from directory: %s" % path)

        data3d_p = os.path.join(path, "data3d.dcm")
        data3d_info_p = os.path.join(path, "data3d.json")
        if not os.path.exists(data3d_p):
            logger.error("Missing file 'data3d.dcm'! Could not load ready data.")
            return
        elif not os.path.exists(data3d_info_p):
            logger.error("Missing file 'data3d.json'! Could not load ready data.")
            return
        data3d, metadata = io3d.datareader.read(data3d_p)
        with open(data3d_info_p, 'r') as fp:
            data3d_info = json.load(fp)

        obj = cls() # to get mask and stats names
        masks = {}; stats = {}

        for part in obj.masks_comp:
            mask_p = os.path.join(path, "%s.dcm" % part)
            if os.path.exists(mask_p):
                tmp, _ = io3d.datareader.read(mask_p)
                masks[part] = compressArray(tmp.astype(np.bool))
                del(tmp)

        for part in obj.stats:
            stats_p = os.path.join(path, "%s.json" % part)
            if os.path.exists(stats_p):
                with open(stats_p, 'r') as fp:
                    tmp = json.load(fp)
                stats[part] = tmp

        return cls.fromReadyData(data3d, data3d_info, masks=masks, stats=stats)

    def toDirectory(self, path):
        """ note: Masks look wierd when opened in ImageJ, but are saved correctly """
        logger.info("Saving all processed data to directory: %s" % path)
        spacing = list(self.spacing)

        data3d_p = os.path.join(path, "data3d.dcm")
        io3d.datawriter.write(self.data3d, data3d_p, 'dcm', {'voxelsize_mm': spacing})

        data3d_info_p = os.path.join(path, "data3d.json")
        data3d_info = {
            "spacing":copy.deepcopy(list(self.spacing)),
            "cut_shape":copy.deepcopy(list(self.cut_shape)),
            "padding":copy.deepcopy(list(self.padding)),
            "orig_voxelsize":copy.deepcopy(list(self.orig_voxelsize)),
            "orig_coord_scale":copy.deepcopy(list(self.orig_coord_scale)),
            "orig_coord_intercept":copy.deepcopy(list(self.orig_coord_intercept))
            }
        with open(data3d_info_p, 'w') as fp:
            json.dump(data3d_info, fp, sort_keys=True)


        for part in self.masks_comp:
            if self.masks_comp[part] is None: continue
            mask_p = os.path.join(path, "%s.dcm" % part)
            mask = self.getPart(part, raw=True).astype(np.int8)
            io3d.datawriter.write(mask, mask_p, 'dcm', {'voxelsize_mm': spacing})
            del(mask)

        for part in self.stats:
            if self.stats[part] is None: continue
            stats_p = os.path.join(path, "%s.json" % part)
            with open(stats_p, 'w') as fp:
                json.dump(self.getStats(part, raw=True), fp, sort_keys=True)

    def prepareData(self, data3d, voxelsize, size_normalization=True, rescale=True):
        """
        Output:
            data3d - prepared data3d
            spacing - normalized voxelsize for computation
            cut_shape - data3d shape before recale and padding
            padding - padding of output data
        """
        logger.info("Preparing input data...")
        # fix for io3d <-512;511> value range bug
        # this is caused by hardcoded slope 0.5 in dcmreader
        if np.min(data3d) >= -512: data3d = data3d * 2

        # limit value range to <-1024;1024>
        # [ data3d < -1024 ] => less dense then air - padding values
        # [ data3d > 1024  ] => more dense then most bones - only teeth (or just CT reaction to teeth fillings)
        data3d[ data3d < -1024 ] = -1024
        data3d[ data3d > 1024 ] = 1024

        # set padding value to -1024
        data3d[ data3d == data3d[0,0,0] ] = -1024

        # <-1024;1024> can fit into int16
        data3d = data3d.astype(np.int16)

        # filter out noise - median filter with radius 1 (kernel 3x3x3)
        data3d = scipy.ndimage.filters.median_filter(data3d, 3)

        # ed = sed3.sed3(data3d); ed.show()

        # remove high brightness errors near edges of valid data (takes about 70s)
        logger.debug("Removing high brightness errors near edges of valid data")
        valid_mask = data3d > -1024
        valid_mask = skimage.measure.label(valid_mask, background=0)
        unique, counts = np.unique(valid_mask, return_counts=True)
        unique = unique[1:]; counts = counts[1:] # remove background label (is 0)
        valid_mask = valid_mask == unique[list(counts).index(max(counts))]
        for z in range(valid_mask.shape[0]):
            tmp = valid_mask[z,:,:]
            if np.sum(tmp) == 0: continue
            tmp = skimage.morphology.convex_hull_image(tmp)
            # get contours
            tmp = (skimage.feature.canny(tmp) != 0)
            # thicken contour (expecting 512x512 resolution)
            tmp = scipy.ndimage.binary_dilation(tmp, structure=skimage.morphology.disk(11, dtype=np.bool))
            # lower all values near border bigger then -300 closer to -300
            dst = scipy.ndimage.morphology.distance_transform_edt(tmp).astype(np.float)
            dst = dst/np.max(dst)
            dst[ dst != 0 ] = 0.01**dst[ dst != 0 ]; dst[ dst == 0 ] = 1.0

            mask = data3d[z,:,:] > -300
            data3d[z,:,:][mask] = ( \
                ((data3d[z,:,:][mask].astype(np.float)+300)*dst[mask])-300 \
                ).astype(np.int16)
        # ed = sed3.sed3(data3d); ed.show()

        # cut off empty parts of data
        logger.debug("Removing array padding")
        body = self._getBody(data3d, voxelsize)
        data3d[ body == 0 ] = -1024
        padding = getDataPadding(body)
        data3d = cropArray(data3d, padding)
        body = cropArray(body, padding)
        cut_shape = data3d.shape # without padding
        # ed = sed3.sed3(data3d); ed.show()

        # size norming based on body size on xy axis
        # this just recalculates voxelsize (doesnt touch actual data)
        if size_normalization:
            # get median widths and heights from just [lungs_end:(lungs_end+200mm),:,:]
            fatlessbody = self._getFatlessBody(data3d, voxelsize, body); del(body)
            lungs = data3d < -300; lungs[ fatlessbody == 0 ] = 0 # very roughly detect end of lungs
            for z in range(data3d.shape[0]):
                if np.sum(lungs[z,:,:]) == 0: continue
                pads = getDataPadding(fatlessbody[z,:,:])
                height = lungs[z,:,:].shape[0]-pads[0][1]-pads[0][0]
                if height == 0: continue
                lungs[z,:int(pads[0][0]+height*(3/4)),:] = 0
            #ed = sed3.sed3(data3d, seeds=lungs); ed.show()
            if np.sum(lungs) == 0:
                lungs_end = 0
            else:
                # use only biggest object
                lungs = skimage.measure.label(lungs, background=0)
                unique, counts = np.unique(lungs, return_counts=True)
                unique = unique[1:]; counts = counts[1:]
                lungs = lungs == unique[list(counts).index(max(counts))]
                # calc idx of last slice with lungs
                lungs_end = data3d.shape[0] - getDataPadding(lungs)[0][1]
            del(lungs)

            widths = []; heights = []
            for z in range(lungs_end, min(int(lungs_end+200/voxelsize[0]), fatlessbody.shape[0])):
                if np.sum(fatlessbody[z,:,:]) == 0: continue
                spads = getDataPadding(fatlessbody[z,:,:])
                heights.append( fatlessbody[z,:,:].shape[0]-np.sum(spads[0]) )
                widths.append( fatlessbody[z,:,:].shape[1]-np.sum(spads[1]) )

            if len(widths) != 0:
                size_v = [ np.median(heights), np.median(widths) ]
            else:
                logger.warning("Could not detect median body (abdomen) width and height! Using size of middle slice for normalization.")
                size_v = [ fatlessbody.shape[dim+1]-np.sum(pad) for dim, pad in enumerate(getDataPadding(fatlessbody[int(fatlessbody.shape[0]/2),:,:])) ]
            del(fatlessbody)

            size_mm = [ size_v[0]*voxelsize[1], size_v[1]*voxelsize[2] ] # fatlessbody size in mm on X and Y axis
            size_scale = [ None, self.NORMED_FATLESS_BODY_SIZE[0]/size_mm[0], self.NORMED_FATLESS_BODY_SIZE[1]/size_mm[1] ]
            size_scale[0] = (size_scale[1]+size_scale[2])/2 # scaling on z-axis is average of scaling on x,y-axis
            new_voxelsize = [
                voxelsize[0]*size_scale[0],
                voxelsize[1]*size_scale[1],
                voxelsize[2]*size_scale[2],
                ]
            logger.debug("Voxelsize normalization: %s -> %s" % (str(voxelsize), str(new_voxelsize)))
            voxelsize = new_voxelsize
        else: del(body) # not needed anymore

        # resize data on x,y axis (upscaling creates ghosting effect on z-axis)
        if not rescale:
            spacing = voxelsize
        else:
            new_shape = np.asarray([ data3d.shape[0], data3d.shape[1] * voxelsize[1], \
                data3d.shape[2] * voxelsize[2] ]).astype(np.int)
            spacing = np.asarray([ voxelsize[0], 1, 1 ])
            logger.debug("Data3D shape resize: %s -> %s; New voxelsize: %s" % (str(data3d.shape), str(tuple(new_shape)), str((voxelsize[0],1,1))))
            data3d = skimage.transform.resize(
                data3d, new_shape, order=3, mode="reflect", clip=True, preserve_range=True,
                ).astype(np.int16)

        # ed = sed3.sed3(data3d); ed.show()
        return data3d, spacing, cut_shape, padding

    def toOutputCoordinates(self, vector, mm=False):
        orig_vector = np.asarray(vector) * self.orig_coord_scale + self.orig_coord_intercept
        if mm: orig_vector = orig_vector * self.orig_voxelsize
        return orig_vector

    def toOutputFormat(self, data, padding_value=0):
        """
        Returns data to the same shape as orginal data used in creation of class object
        """
        out = skimage.transform.resize(
            data, self.cut_shape, order=3, mode="reflect", clip=True, preserve_range=True
            )
        if data.dtype in [np.bool,np.int,np.uint]:
            out = np.round(out).astype(data.dtype)
        out = padArray(out, self.padding, padding_value=padding_value)
        return out

    def getData3D(self, raw=False):
        data = self.data3d
        if not raw: data = self.toOutputFormat(data, padding_value=-1024)
        return data.copy()

    def getPart(self, part, raw=False):
        part = part.strip().lower()

        if part not in self.masks_comp:
            logger.error("Invalid bodypart '%s'! Returning empty mask!" % part)
            data = np.zeros(self.data3d.shape)

        elif self.masks_comp[part] is not None:
            data = decompressArray(self.masks_comp[part])

        else:
            if part == "body":
                data = self._getBody(self.data3d, self.spacing)
            elif part == "fatlessbody":
                data = self._getFatlessBody(self.data3d, self.spacing, self.getBody(raw=True))
            elif part == "lungs":
                data = self._getLungs(self.data3d, self.spacing, self.getFatlessBody(raw=True))
            elif part == "diaphragm":
                data = self._getDiaphragm(self.data3d, self.spacing, self.getLungs(raw=True))
            elif part == "bones":
                data = self._getBones(self.data3d, self.spacing, self.getFatlessBody(raw=True), \
                    self.getLungs(raw=True) )
            elif part == "abdomen":
                data = self._getAbdomen(self.data3d, self.spacing, self.getFatlessBody(raw=True), \
                    self.getDiaphragm(raw=True), self.analyzeBones(raw=True))
            elif part == "vessels":
                data = self._getVessels(self.data3d, self.spacing, \
                    self.getBones(raw=True), self.analyzeBones(raw=True) )
            elif part == "aorta":
                data = self._getAorta(self.data3d, self.spacing, self.getVessels(raw=True), \
                    self.analyzeVessels(raw=True) )
            elif part == "venacava":
                data = self._getVenaCava(self.data3d, self.spacing, self.getVessels(raw=True), \
                    self.analyzeVessels(raw=True) )

            self.masks_comp[part] = compressArray(data)

        if not raw: data = self.toOutputFormat(data)
        return data

    def getBody(self, raw=False):
        return self.getPart("body", raw=raw)

    def getFatlessBody(self, raw=False):
        return self.getPart("fatlessbody", raw=raw)

    def getLungs(self, raw=False):
        return self.getPart("lungs", raw=raw)

    def getDiaphragm(self, raw=False):
        return self.getPart("diaphragm", raw=raw)

    def getBones(self, raw=False):
        return self.getPart("bones", raw=raw)

    def getAbdomen(self, raw=False):
        return self.getPart("abdomen", raw=raw)

    def getVessels(self, raw=False):
        return self.getPart("vessels", raw=raw)

    def getAorta(self, raw=False):
        return self.getPart("aorta", raw=raw)

    def getVenaCava(self, raw=False):
        return self.getPart("venacava", raw=raw)

    #
    # Segmentation algorithms
    #

    @classmethod
    def _getBody(cls, data3d, spacing):
        """
        Input: noiseless data3d
        Returns binary mask representing body volume (including most cavities)
        """
        logger.info("_getBody")
        # segmentation of body volume
        body = (data3d > -300).astype(np.bool)

        # fill holes
        body = binaryFillHoles(body, z_axis=True, y_axis=True, x_axis=True)

        # binary opening
        body = scipy.ndimage.morphology.binary_opening(body, structure=getSphericalMask([5,]*3, spacing=spacing))

        # leave only biggest object in data
        body_label = skimage.measure.label(body, background=0)
        unique, counts = np.unique(body_label, return_counts=True)
        unique = unique[1:]; counts = counts[1:] # remove background label (is 0)
        body = body_label == unique[list(counts).index(max(counts))]

        # filling nose/mouth openings + connected cavities
        # - fills holes separately on every slice along z axis (only part of mouth and nose should have cavity left)
        for z in range(body.shape[0]):
            body[z,:,:] = binaryFillHoles(body[z,:,:])

        return body

    @classmethod
    def _getFatlessBody(cls, data3d, spacing, body): # TODO - ignore nipples when creating convex hull
        """
        Returns convex hull of body without fat and skin
        """
        logger.info("_getFatlessBody")
        # remove fat
        fatless = (data3d > 20)
        fatless = scipy.ndimage.morphology.binary_opening(fatless, structure=np.ones([3,3,3])) # small remove segmentation errors
        # fill body cavities, but ignore air near borders of body
        body_border = body & ( scipy.ndimage.morphology.binary_erosion(body, \
            structure=np.expand_dims(skimage.morphology.disk(9, dtype=np.bool), axis=0)) == 0)
        fatless[ (data3d < -300) & (body_border == 0) & (body == 1) ] = 1
        # remove skin
        tmp = scipy.ndimage.morphology.binary_opening(fatless, structure=getSphericalMask([7,7,7], spacing=spacing))
        fatless[ body_border ] = tmp[ body_border ]
        #ed = sed3.sed3(data3d, contour=fatless, seeds=body_border); ed.show()
        # save convex hull along z-axis
        for z in range(fatless.shape[0]):
            bsl = skimage.measure.label(body[z,:,:], background=0)
            for l in np.unique(bsl)[1:]:
                tmp = fatless[z,:,:] & (bsl == l)
                if np.sum(tmp) == 0: continue
                fatless[z,:,:][ skimage.morphology.convex_hull_image(tmp) == 1 ] = 1
                fatless[z,:,:][ body[z,:,:] == 0 ] = 0
        return fatless

    @classmethod
    def _getLungs(cls, data3d, spacing, fatlessbody): # TODO - check how much memory this eats
        """ Expects lungs to actually be in data """
        logger.info("_getLungs")
        lungs = data3d < -300
        lungs[ fatlessbody == 0 ] = 0
        lungs = binaryFillHoles(lungs, z_axis=True)

        # remove all blobs that don't go through lower 1/4 of body
        lungs = skimage.measure.label(lungs, background=0)
        valid_labels = []
        for z in range(data3d.shape[0]):
            if np.sum(lungs[z,:,:]) == 0: continue
            pads = getDataPadding(fatlessbody[z,:,:])
            height = lungs[z,:,:].shape[0]-pads[0][1]-pads[0][0]
            if height == 0: continue
            unique = np.unique(lungs[z,int(pads[0][0]+height*(3/4)):,:])[1:]
            for u in unique:
                if u not in valid_labels:
                    valid_labels.append(u)
        for u in valid_labels:
            lungs[ lungs == u ] = -1
        lungs = lungs == -1
        #ed = sed3.sed3(data3d, contour=lungs); ed.show()

        # try to separate connected intestines
        wseeds = np.zeros(data3d.shape, dtype=np.uint8)
        for z in range(data3d.shape[0]):
            if np.sum(lungs[z,:,:]) == 0: continue
            pads = getDataPadding(fatlessbody[z,:,:])
            height = lungs[z,:,:].shape[0]-pads[0][1]-pads[0][0]
            # cavities that are in lower 1/3 of slice are lungs
            wseeds[z,int(pads[0][0]+height*(2/3)):,:] = 1
            wseeds[z, lungs[z,:,:] == 0 ] = 0
            # slices that have cavities only in upper 2/3 of slice are intestines
            # - to ignore trachea slices, use only lower half of z-axis
            # - do not put any seeds in slices transitioning from lungs to intestines
            if (z > data3d.shape[0]/2) and (np.sum(wseeds[max(0,z-5):(z+1),:,:] == 1) == 0) and \
                (np.sum(lungs[z,int(pads[0][0]+height*(2/3)):,:]) == 0):
                    wseeds[z,:int(pads[0][0]+height*(2/3)),:] = 2
                    wseeds[z, lungs[z,:,:] == 0 ] = 0
            # grow seeds into segmented objects in this slice
            if np.sum(wseeds[z,:,:]) == 0: continue
            slbl = skimage.measure.label(lungs[z,:,:], background=0)
            unique = np.unique(slbl)[1:]
            for u in unique:
                s = np.unique(wseeds[z, slbl == u ])[1:]
                if len(s) == 1:
                    wseeds[z, slbl == u ] = s[0]
        #ed = sed3.sed3(data3d, contour=lungs, seeds=wseeds); ed.show()
        lungs = skimage.morphology.watershed(lungs, wseeds, mask=lungs)
        #ed = sed3.sed3(data3d, contour=lungs, seeds=wseeds); ed.show()
        lungs = lungs == 1

        # leave only lungs in data (1st and 2nd biggest objects, with similar centroids)
        lungs = skimage.measure.label(lungs, background=0)
        #ed = sed3.sed3(lungs); ed.show()
        unique, counts = np.unique(lungs, return_counts=True)
        unique = unique[1:]; counts = counts[1:] # remove background label (is 0)
        centroids = scipy.ndimage.measurements.center_of_mass(lungs, lungs, unique)
        if len(unique) == 0:
            logger.warning("Couldn't find lungs!")
            return np.zeros(data3d.shape, dtype=np.bool)

        idx_1st = list(counts).index(max(counts))
        count_1st = counts[idx_1st]
        centroid_1st = centroids[idx_1st]

        idx_2nd = None
        count_2nd = 0
        centroid_2nd = None
        if len(unique) >= 2:
            counts[idx_1st] = 0
            idx_2nd = list(counts).index(max(counts))
            count_2nd = counts[idx_2nd]
            counts[idx_1st] = count_1st
            centroid_2nd = centroids[idx_2nd]

        #print(count_1st, count_2nd)
        lungs[ lungs == unique[idx_1st] ] = -1
        if len(unique) >= 2:
            ok = True
            # if second biggest is not at least 40% as big as first -> bad
            if count_2nd/count_1st < 0.4: ok = False
            # if centroids are too distant at z and y axis -> bad
            if (centroid_1st[0]-centroid_2nd[0])*spacing[0] > 50: ok = False
            if (centroid_1st[1]-centroid_2nd[1])*spacing[1] > 50: ok = False

            if ok: lungs[ lungs == unique[idx_2nd] ] = -1
        lungs = lungs == -1

        # remove trachea (only the part sticking out)
        pads = getDataPadding(lungs)
        lungs_depth_mm = (lungs.shape[0]-pads[0][1]-pads[0][0])*spacing[0]
        if lungs_depth_mm > 200: # if lungs are longer then 200 mm on z-axis -> trying to remove trachea should not lungs from abdomen-only data
            pads = getDataPadding(lungs)
            s = ( slice(pads[0][0],lungs.shape[0]-pads[0][1]), \
                slice(pads[1][0],lungs.shape[1]-pads[1][1]), \
                slice(pads[2][0],lungs.shape[2]-pads[2][1]) )
            tmp = lungs.copy()
            tmp[s] = binaryClosing(tmp[s], structure=getSphericalMask([10,]*3, spacing=spacing))
            tmp[s] = scipy.ndimage.morphology.binary_opening(tmp[s], \
                structure=getSphericalMask([30,]*3, spacing=spacing))
            lungs[:getDataPadding(tmp)[0][0],:,:] = 0

        return lungs

    @classmethod
    def _getDiaphragm(cls, data3d, spacing, lungs):
        """ Returns interpolated shape of Thoracic diaphragm (continues outsize of body) """
        logger.info("_getDiaphragm")
        diaphragm = scipy.ndimage.filters.sobel(lungs.astype(np.int16), axis=0) < -10

        # create diaphragm heightmap
        heightmap = np.zeros((diaphragm.shape[1], diaphragm.shape[2]), dtype=np.float)
        lungs_stop = lungs.shape[0]-getDataPadding(lungs)[0][1]
        diaphragm_start = max(0, lungs_stop - int(100/spacing[0]))
        for y in range(diaphragm.shape[1]):
            for x in range(diaphragm.shape[2]):
                if np.sum(diaphragm[:,y,x]) == 0:
                    heightmap[y,x] = np.nan
                else:
                    tmp = diaphragm[:,y,x][::-1]
                    z = len(tmp) - np.argmax(tmp) - 1
                    if z < diaphragm_start:
                        # make sure that diaphragm is not higher then lowest lungs point -100mm
                        heightmap[y,x] = np.nan
                    else:
                        heightmap[y,x] = z

        # interpolate missing values
        height_median = np.nanmedian(heightmap)
        x = np.arange(0, heightmap.shape[1])
        y = np.arange(0, heightmap.shape[0])
        heightmap = np.ma.masked_invalid(heightmap)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~heightmap.mask]
        y1 = yy[~heightmap.mask]
        newarr = heightmap[~heightmap.mask]
        heightmap = scipy.interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), \
            method='linear', fill_value=height_median)
        #ed = sed3.sed3(np.expand_dims(heightmap, axis=0)); ed.show()

        # 2D heightmap -> 3D diaphragm
        diaphragm = np.zeros(diaphragm.shape, dtype=np.bool)
        for y in range(diaphragm.shape[1]):
            for x in range(diaphragm.shape[2]):
                z = int(heightmap[y,x])
                diaphragm[z,y,x] = 1

        # make sure that diaphragm is lower then lungs volume
        diaphragm[ lungs == 1 ] = 1
        for y in range(diaphragm.shape[1]):
            for x in range(diaphragm.shape[2]):
                tmp = diaphragm[:,y,x][::-1]
                z = len(tmp) - np.argmax(tmp) - 1
                diaphragm[:,y,x] = 0
                diaphragm[z,y,x] = 1

        #ed = sed3.sed3(data3d, seeds=diaphragm); ed.show()
        return diaphragm

    @classmethod
    def _getBones(cls, data3d, spacing, fatless, lungs):
        """
        Good enough sgementation of all bones
        * data3d - everything, but body must be removed
        """
        logger.info("_getBones")
        spacing_vol = spacing[0]*spacing[1]*spacing[2]
        fatless_dst = scipy.ndimage.morphology.distance_transform_edt(fatless, sampling=spacing)

        # get voxels that are mostly bones
        bones = (data3d > 300).astype(np.bool)
        bones = binaryFillHoles(bones, z_axis=True)
        bones = skimage.morphology.remove_small_objects(bones.astype(np.bool), min_size=int((10**3)/spacing_vol))
        # readd segmented points that are in expected ribs volume
        bones[ (fatless_dst < 15) & (fatless == 1) & (data3d > 300) ] = 1

        # remove possible segmented heart parts (remove upper half of convex hull of lungs)
        #ed = sed3.sed3(data3d, contour=lungs); ed.show()
        if np.sum(lungs) != 0:
            pads = getDataPadding(lungs)
            s = ( slice(pads[0][0],data3d.shape[0]-pads[0][1]), \
                slice(pads[1][0],data3d.shape[1]-pads[1][1]), \
                slice(pads[2][0],data3d.shape[2]-pads[2][1]) )
            lungs_hull = lungs[s]
            for z in range(lungs_hull.shape[0]):
                lungs_hull[z,:,:] = skimage.morphology.convex_hull_image(lungs_hull[z,:,:])
            bones[s][:,:int(lungs_hull.shape[1]/2),:][ lungs_hull[:,:int(lungs_hull.shape[1]/2),:] == 1 ] = 0

        #ed = sed3.sed3(data3d, contour=bones); ed.show()

        # segmentation > 200, save only objects that are connected from > 300
        b200 = skimage.measure.label((data3d > 200), background=0)
        seeds_l = b200.copy(); seeds_l[ bones == 0 ] = 0
        for l in np.unique(seeds_l)[1:]:
            b200[ b200 == l ] = -1
        b200 = (b200 == -1); del(seeds_l)

        # remove stuff connected to heart
        if np.sum(lungs) != 0:
            wseeds = ( bones == 1 ).astype(np.int8) # = 1
            wseeds[ (fatless_dst < 15) & (fatless == 1) & (b200 == 1) ] = 1 # ribs readded
            wseeds[s][:,:int(lungs_hull.shape[1]/2),:][ lungs_hull[:,:int(lungs_hull.shape[1]/2),:] == 1 ] = 2
            b200 = skimage.morphology.watershed(b200, wseeds, mask=b200) == 1

            #ed = sed3.sed3(data3d, seeds=wseeds, contour=r); ed.show()

            # again remove all not connected to > 300
            b200 = skimage.measure.label(b200, background=0)
            seeds_l = b200.copy(); seeds_l[ bones == 0 ] = 0
            for l in np.unique(seeds_l)[1:]:
                b200[ b200 == l ] = -1
            b200 = (b200 == -1); del(seeds_l)

        bones = b200; del(b200)
        bones = binaryClosing(bones, structure=getSphericalMask([5,]*3, spacing=spacing))
        bones = binaryFillHoles(bones, z_axis=True)

        #ed = sed3.sed3(data3d, contour=bones); ed.show()

        return bones

    @classmethod
    def _getAbdomen(cls, data3d, spacing, fatlessbody, diaphragm, bones_stats):
        """ Helpful for segmentation of organs in abdomen """
        logger.info("_getAbdomen")
        # define abdomen as fatless volume under diaphragm
        abdomen = fatlessbody.copy()
        for y in range(diaphragm.shape[1]):
            for x in range(diaphragm.shape[2]):
                tmp = diaphragm[:,y,x][::-1]
                z = len(tmp) - np.argmax(tmp) - 1
                abdomen[:z+1,y,x] = 0

        # remove everything under hip joints
        if len(bones_stats["hip_joints"]) != 0:
            hips_start = bones_stats["hip_joints"][0][0]
            abdomen[hips_start:,:,:] = 0

        #ed = sed3.sed3(data3d, contour=abdomen); ed.show()
        return abdomen

    @classmethod
    def _getVessels(cls, data3d, spacing, bones, bones_stats, contrast_agent=True):
        """
        Tabular value of blood radiodensity is 13-50 HU.
        When contrast agent is used, it rises to roughly 100-140 HU.
        Vessels are segmentable only if contrast agent was used.
        """
        logger.info("_getVessels")
        points_spine = bones_stats["spine"]
        if len(points_spine) == 0:
            logger.warning("Couldn't find vessels!")
            return np.zeros(data3d.shape, dtype=np.bool)
        # get spine z-range
        spine_zmin = points_spine[0][0]; spine_zmax = points_spine[-1][0]

        #ed = sed3.sed3(data3d, contour=bones); ed.show()

        VESSEL_THRESHOLD = 110 # 145
        SPINE_WIDTH = int(22/spacing[2]) # from center
        SPINE_HEIGHT = int(30/spacing[1]) # from center

        if contrast_agent:
            vessels = data3d > VESSEL_THRESHOLD

            wseeds = bones.astype(np.uint8) # = 1
            for z in range(spine_zmin,spine_zmax+1): # draw seeds elipse at spine center
                sc = points_spine[z-spine_zmin]; sc = (sc[1], sc[2])
                rr, cc = skimage.draw.ellipse(sc[0], sc[1], SPINE_HEIGHT, SPINE_WIDTH, \
                    shape=wseeds[z,:,:].shape)
                wseeds[z,rr,cc] = 1
            wseeds[ scipy.ndimage.morphology.distance_transform_edt(wseeds == 0, sampling=spacing) > 15 ] = 2
            wseeds[ vessels == 0 ] = 0 # seeds only where there are vessels

            vessels = skimage.morphology.watershed(vessels, wseeds, mask=vessels)
            #ed = sed3.sed3(data3d, seeds=wseeds, contour=vessels); ed.show()
            vessels = vessels == 2 # even smallest vessels and kidneys

            vessels = scipy.ndimage.morphology.binary_fill_holes(vessels)
            vessels = scipy.ndimage.binary_opening(vessels, structure=np.ones((3,3,3)))

            # remove vessels outside of detected spine z-range
            vessels[:spine_zmin,:,:] = 0
            vessels[spine_zmax+1:,:,:] = 0
            #ed = sed3.sed3(data3d, contour=vessels); ed.show()

            # remove liver and similar half-segmented organs
            cut_rad = (150, 70); cut_rad = (cut_rad[0]/spacing[1], cut_rad[1]/spacing[2])
            wseeds = np.zeros(vessels.shape, dtype=np.int8)
            for z in range(spine_zmin,spine_zmax+1):
                vs = vessels[z,:,:]; sc = points_spine[z-spine_zmin]; sc = (sc[1], sc[2])

                rr, cc = skimage.draw.ellipse(sc[0]-cut_rad[0]-SPINE_HEIGHT, sc[1], \
                    cut_rad[0], cut_rad[1], shape=wseeds[z,:,:].shape)
                wseeds[z,rr,cc] = 1

                rr, cc = skimage.draw.ellipse(sc[0], sc[1], cut_rad[0], cut_rad[1], \
                    shape=wseeds[z,:,:].shape)
                mask = np.zeros(wseeds[z,:,:].shape); mask[rr, cc] = 1
                mask[int(sc[0]):,:] = 0
                wseeds[z, mask != 1] = 2
            # ed = sed3.sed3(data3d, seeds=wseeds, contour=vessels); ed.show()

            r = skimage.morphology.watershed(vessels, wseeds, mask=vessels)
            vessels = r == 1
            #ed = sed3.sed3(data3d, contour=vessels); ed.show()

            # find circles near spine
            rad = np.asarray(list(range(9,12)), dtype=np.float32)
            rad = list( rad / float((spacing[1]+spacing[2])/2.0) )
            wseeds = np.zeros(vessels.shape, dtype=np.int8)
            for z in range(spine_zmin,spine_zmax+1):
                vs = vessels[z,:,:]; sc = points_spine[z-spine_zmin]; sc = (sc[1], sc[2])
                spine_height = sc[0]-SPINE_HEIGHT

                # get circle centers
                edge = skimage.feature.canny(vs, sigma=0.0)
                #ed = sed3.sed3(np.expand_dims(edge.astype(np.float), axis=0)); ed.show()
                r = skimage.transform.hough_circle(edge, radius=rad)
                #ed = sed3.sed3(r, contour=np.expand_dims(vs, axis=0)); ed.show()
                r = np.sum(r > 0.35, axis=0) != 0
                r[ vs == 0 ] = 0 # remove centers outside segmented vessels
                r = scipy.ndimage.binary_closing(r, structure=np.ones((10,10))) # connect near centers
                #ed = sed3.sed3(np.expand_dims(r.astype(np.float), axis=0), contour=np.expand_dims(vs, axis=0)); ed.show()

                # get circle centers
                if np.sum(r) == 0: continue
                rl = skimage.measure.label(r, background=0)
                centers = scipy.ndimage.measurements.center_of_mass(r, rl, range(1, np.max(rl)+1))

                # use only circle centers that are near spine, and are in vessels
                for i, c in enumerate(centers):
                    dst_y = abs(sc[0]*spacing[1]-c[0]*spacing[1])
                    dst_x = abs(sc[1]*spacing[2]-c[1]*spacing[2])
                    dst2 = dst_y**2 + dst_x**2
                    if vs[int(c[0]),int(c[1])] == 0: continue # must be inside segmented vessels
                    elif dst2 > 70**2: continue # max dist from spine
                    elif c[0] > spine_height: continue # no lower then spine height
                    else: wseeds[z,int(c[0]),int(c[1])] = 1

            # convolution with vertical kernel to remove seeds in vessels not going up-down
            kernel = np.ones((15,1,1))
            r = scipy.ndimage.convolve(vessels.astype(np.uint32), kernel)
            #ed = sed3.sed3(r, contour=vessels); ed.show()
            wseeds[ r < np.sum(kernel) ] = 0

            # remove everything thats not connected to at least one seed
            vessels = skimage.measure.label(vessels, background=0)
            tmp = vessels.copy(); tmp[ wseeds == 0 ] = 0
            for l in np.unique(tmp)[1:]:
                vessels[ vessels == l ] = -1
            vessels = (vessels == -1); del(tmp)

            # watershed
            wseeds_base = wseeds.copy() # only circle centers
            wseeds = scipy.ndimage.binary_dilation(wseeds.astype(np.bool), structure=np.ones((1,3,3))).astype(np.int8)
            cut_rad = (90, 70); cut_rad = (cut_rad[0]/spacing[1], cut_rad[1]/spacing[2])
            for z in range(spine_zmin,spine_zmax+1):
                sc = points_spine[z-spine_zmin]; sc = (sc[1], sc[2])
                rr, cc = skimage.draw.ellipse(sc[0], sc[1], cut_rad[0], cut_rad[1], shape=wseeds[z,:,:].shape)
                mask = np.zeros(wseeds[z,:,:].shape); mask[rr, cc] = 1
                mask[int(sc[0]):,:] = 0
                wseeds[z, mask != 1] = 2
            #ed = sed3.sed3(data3d, seeds=wseeds, contour=vessels); ed.show()

            r = skimage.morphology.watershed(vessels, wseeds, mask=vessels)
            #ed = sed3.sed3(data3d, seeds=wseeds, contour=r); ed.show()
            vessels = r == 1

            # remove everything thats not connected to at least one seed, again (just to be safe)
            vessels = skimage.measure.label(vessels, background=0)
            tmp = vessels.copy(); tmp[ wseeds_base == 0 ] = 0
            for l in np.unique(tmp)[1:]:
                vessels[ vessels == l ] = -1
            vessels = (vessels == -1); del(tmp)

            return vessels

        else: # without contrast agent, blood is 13-50 HU
            logger.warning("Couldn't find vessels!")
            return np.zeros(data3d.shape, dtype=np.bool)

            # TODO - try it anyway
            # - FIND EDGES, THRESHOLD EDGES - canny
            # - hough_circle
            # - convolution with kernel with very big z-axis
            # - combine last two steps
            # - points_spine - select close circles to spine

    @classmethod
    def _getAorta(cls, data3d, spacing, vessels, vessels_stats):
        logger.info("_getAorta")
        points = vessels_stats["aorta"]
        if len(points) == 0 or np.sum(vessels) == 0:
            logger.warning("Couldn't find aorta volume!")
            return np.zeros(vessels.shape, dtype=np.bool)

        VESSEL_RADIUS = 12

        aorta = np.zeros(vessels.shape, dtype=np.bool)
        for p in points:
            aorta[p[0],p[1],p[2]] = 1
        aorta = growRegion(aorta, vessels, iterations=VESSEL_RADIUS)

        return aorta

    @classmethod
    def _getVenaCava(cls, data3d, spacing, vessels, vessels_stats):
        logger.info("_getVenaCava")
        points = vessels_stats["vena_cava"]
        if len(points) == 0 or np.sum(vessels) == 0:
            logger.warning("Couldn't find venacava volume!")
            return np.zeros(vessels.shape, dtype=np.bool)

        VESSEL_RADIUS = 12

        venacava = np.zeros(vessels.shape, dtype=np.bool)
        for p in points:
            venacava[p[0],p[1],p[2]] = 1
        venacava = growRegion(venacava, vessels, iterations=VESSEL_RADIUS)

        return venacava

    ################

    def getStats(self, part, raw=False):
        part = part.strip().lower()

        if part not in self.stats:
            logger.error("Invalid stats bodypart '%s'! Returning empty dictionary!" % part)
            data = {}

        elif self.stats[part] is not None:
            data = copy.deepcopy(self.stats[part])

        else:
            if part == "bones":
                data = self._analyzeBones( \
                data3d=self.data3d, spacing=self.spacing, fatlessbody=self.getFatlessBody(raw=True), \
                bones=self.getBones(raw=True), lungs=self.getLungs(raw=True) )
            elif part == "vessels":
                data = self._analyzeVessels( \
                data3d=self.data3d, spacing=self.spacing, vessels=self.getVessels(raw=True), \
                bones_stats=self.analyzeBones(raw=True) )

            self.stats[part] = copy.deepcopy(data)

        if not raw:
            if part == "bones":
                data["spine"] = [ tuple(self.toOutputCoordinates(p).astype(np.int)) for p in data["spine"] ]
                data["hip_joints"] = [ tuple(self.toOutputCoordinates(p).astype(np.int)) for p in data["hip_joints"] ]
            elif part == "vessels":
                data["aorta"] = [ tuple(self.toOutputCoordinates(p).astype(np.int)) for p in data["aorta"] ]
                data["vena_cava"] = [ tuple(self.toOutputCoordinates(p).astype(np.int)) for p in data["vena_cava"] ]
        return data

    def analyzeBones(self, raw=False):
        return self.getStats("bones", raw=raw)

    def analyzeVessels(self, raw=False):
        return self.getStats("vessels", raw=raw)

    @classmethod
    def _analyzeBones(cls, data3d, spacing, fatlessbody, bones, lungs):
        """ Returns: {"spine":points_spine, "hip_joints":points_hip_joints} """
        logger.info("_analyzeBones")

        # remove every bone higher then lungs
        lungs_pad = getDataPadding(lungs)
        lungs_start = lungs_pad[0][0] # start of lungs on z-axis
        lungs_end = lungs.shape[0]-lungs_pad[0][1] # end of lungs on z-axis
        bones[:lungs_start,:,:] = 0 # definitely not spine or hips
        for z in range(0, lungs_end): # remove front parts of ribs (to get correct spine center)
            bs = fatlessbody[z,:,:]; pad = getDataPadding(bs)
            height = int(bones.shape[1]-(pad[1][0]+pad[1][1]))
            top_sep = pad[1][0]+int(height*0.3)
            bones[z,:top_sep,:] = 0

        # merge near "bones" into big blobs
        bones[lungs_start:,:,:] = binaryClosing(bones[lungs_start:,:,:], \
            structure=getSphericalMask([20,]*3, spacing=spacing)) # takes around 1m

        #ed = sed3.sed3(data3d, contour=bones); ed.show()

        points_spine = []
        points_hip_joints_l = []; points_hip_joints_r = []
        for z in range(lungs_start, bones.shape[0]): # TODO - separate into more sections (spine should be only in middle-lower)
            bs = fatlessbody[z,:,:]
            # separate body/bones into 3 sections (on x-axis)
            pad = getDataPadding(bs)
            width = bs.shape[1]-(pad[1][0]+pad[1][1])
            left_sep = pad[1][0]+int(width*0.35)
            right_sep = bs.shape[1]-(pad[1][1]+int(width*0.35))
            left = bones[z,:,pad[1][0]:left_sep]
            center = bones[z,:,left_sep:right_sep]
            right = bones[z,:,right_sep:(bs.shape[1]-pad[1][1])]
            # calc centers and volumes
            left_v = np.sum(left); center_v = np.sum(center); right_v = np.sum(right)
            total_v = left_v+center_v+right_v
            if total_v == 0: continue

            left_c = list(scipy.ndimage.measurements.center_of_mass(left))
            left_c[1] = left_c[1]+pad[1][0]
            center_c = list(scipy.ndimage.measurements.center_of_mass(center))
            center_c[1] = center_c[1]+left_sep
            right_c  = list(scipy.ndimage.measurements.center_of_mass(right))
            right_c[1] = right_c[1]+right_sep

            # try to detect spine center
            if ((left_v/total_v < 0.2) or (right_v/total_v < 0.2)) and (center_v != 0):
                points_spine.append( (z, int(center_c[0]), int(center_c[1])) )

            # try to detect hip joints
            if (z >= lungs_end) and (left_v/total_v > 0.4) and (right_v/total_v > 0.4):
                # gets also leg bones
                #print(z, abs(left_c[1]-right_c[1]))
                if abs(left_c[1]-right_c[1]) < (180.0/spacing[2]): # max hip dist. 180mm
                    # anything futher out should be only leg bones
                    points_hip_joints_l.append( (z, int(left_c[0]), int(left_c[1])) )
                    points_hip_joints_r.append( (z, int(right_c[0]), int(right_c[1])) )

        # calculate centroid of hip points
        points_hip_joints = []
        if len(points_hip_joints_l) != 0:
            z, y, x = zip(*points_hip_joints_l); l = len(z)
            cl = (int(sum(z)/l), int(sum(y)/l), int(sum(x)/l))
            z, y, x = zip(*points_hip_joints_r); l = len(z)
            cr = (int(sum(z)/l), int(sum(y)/l), int(sum(x)/l))
            points_hip_joints = [cl, cr]

        # remove any spine points under detected hips
        if len(points_hip_joints) != 0:
            newp = []
            for p in points_spine:
                if p[0] < points_hip_joints[0][0]:
                    newp.append(p)
            points_spine = newp

        # fit curve to spine points and recalculate new points from curve
        if len(points_spine) >= 2:
            points_spine = polyfit3D(points_spine)

        # seeds = np.zeros(bones.shape)
        # for p in points_spine_c: seeds[p[0], p[1], p[2]] = 2
        # for p in points_spine: seeds[p[0], p[1], p[2]] = 1
        # for p in points_hip_joints_l: seeds[p[0], p[1], p[2]] = 2
        # for p in points_hip_joints_r: seeds[p[0], p[1], p[2]] = 2
        # for p in points_hip_joints: seeds[p[0], p[1], p[2]] = 3
        # seeds = scipy.ndimage.morphology.grey_dilation(seeds, size=(1,5,5))
        # ed = sed3.sed3(data3d, contour=bones, seeds=seeds); ed.show()

        return {"spine":points_spine, "hip_joints":points_hip_joints}

    @classmethod
    def _analyzeVessels(cls, data3d, spacing, vessels, bones_stats):
        """ Returns: {"aorta":[], "vena_cava":[]} """
        logger.info("_analyzeVessels")
        if np.sum(vessels) == 0:
            logger.warning("No vessels to find vessel points for!")
            return {"aorta":[], "vena_cava":[]}

        points_spine = bones_stats["spine"]
        spine_zmin = points_spine[0][0]; spine_zmax = points_spine[-1][0]
        rad = np.asarray([ 7,8,9,10,11,12,13,14 ], dtype=np.float32)
        rad = list( rad / float((spacing[1]+spacing[2])/2.0) )

        points_aorta = []; points_vena_cava = []; points_unknown = [];
        for z in range(spine_zmin,spine_zmax+1): # TODO - ignore space around heart (aorta), start under heart (vena cava)
            sc = points_spine[z-spine_zmin]; sc = (sc[1], sc[2])
            vs = vessels[z,:,:]

            edge = skimage.feature.canny(vs, sigma=0.0)
            r = skimage.transform.hough_circle(edge, radius=rad) > 0.4
            r = np.sum(r, axis=0) != 0
            r[ vs == 0 ] = 0 # remove centers outside segmented vessels
            r = scipy.ndimage.binary_closing(r, structure=np.ones((10,10))) # connect near centers

            # get circle centers
            if np.sum(r) == 0: continue
            rl = skimage.measure.label(r, background=0)
            centers = scipy.ndimage.measurements.center_of_mass(r, rl, range(1, np.max(rl)+1))

            # sort points between aorta, vena_cava and unknown
            for c in centers:
                c_zyx = (z, int(c[0]), int(c[1]))
                # spine center -> 100% aorta
                if sc[1] < c[1]: points_aorta.append(c_zyx)
                # 100% venacava <- spine center - a bit more
                elif c[1] < (sc[1]-20/spacing[2]) : points_vena_cava.append(c_zyx)
                else: points_unknown.append(c_zyx)

        # use watershed find where unknown points are
        cseeds = np.zeros(vessels.shape, dtype=np.int8)
        for p in points_aorta:
            cseeds[p[0],p[1],p[2]] = 1
        for p in points_vena_cava:
            cseeds[p[0],p[1],p[2]] = 2
        r = skimage.morphology.watershed(vessels, cseeds, mask=vessels)
        #ed = sed3.sed3(data3d, contour=r, seeds=cseeds); ed.show()

        for p in points_unknown:
            if r[p[0],p[1],p[2]] == 1:
                points_aorta.append(p)
            elif r[p[0],p[1],p[2]] == 2:
                points_vena_cava.append(p)

        # sort points by z coordinate
        points_aorta = sorted(points_aorta, key=itemgetter(0))
        points_vena_cava = sorted(points_vena_cava, key=itemgetter(0))

        # try to remove outliners, only one point per z-axis slice
        # use points closest to spine # TODO - make this better
        if len(points_aorta) >= 1:
            points_aorta_new = []
            for z, pset in groupby(points_aorta, key=itemgetter(0)):
                pset = list(pset)
                if len(pset) == 1:
                    points_aorta_new.append(pset[0])
                else:
                    sc = points_spine[z-spine_zmin]
                    dists = [ ((p[1]-sc[1])**2 + (p[2]-sc[2])**2) for p in pset ]
                    points_aorta_new.append(pset[list(dists).index(min(dists))])
            points_aorta = points_aorta_new
        if len(points_vena_cava) >= 1:
            points_vena_cava_new = []
            for z, pset in groupby(points_vena_cava, key=itemgetter(0)):
                pset = list(pset)
                if len(pset) == 1:
                    points_vena_cava_new.append(pset[0])
                else:
                    sc = points_spine[z-spine_zmin]
                    dists = [ ((p[1]-sc[1])**2 + (p[2]-sc[2])**2) for p in pset ]
                    points_vena_cava_new.append(pset[list(dists).index(min(dists))])
            points_vena_cava = points_vena_cava_new

        # polyfit curve
        if len(points_aorta) >= 2:
            points_aorta = polyfit3D(points_aorta)
        if len(points_vena_cava) >= 2:
            points_vena_cava = polyfit3D(points_vena_cava)

        return {"aorta":points_aorta, "vena_cava":points_vena_cava}

if __name__ == "__main__":
    import argparse

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # input parser
    parser = argparse.ArgumentParser(description="Organ Detection")
    parser.add_argument('-i','--datadir', default=None,
            help='path to data dir')
    parser.add_argument('-r','--readydir', default=None,
            help='path to ready data dir (for testing)')
    parser.add_argument("--dump", default=None,
            help='dump all processed data to path and exit')
    parser.add_argument("-d", "--debug", action="store_true",
            help='run in debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if (args.datadir is None) and (args.readydir is None):
        logger.error("Missing data directory path --datadir or --readydir")
        sys.exit(2)
    elif (args.datadir is not None) and (not os.path.exists(args.datadir)):
        logger.error("Invalid data directory path --datadir")
        sys.exit(2)
    elif (args.readydir is not None) and (not os.path.exists(args.readydir)):
        logger.error("Invalid data directory path --readydir")
        sys.exit(2)

    if args.datadir is not None:
        print("Reading DICOM dir...")
        data3d, metadata = io3d.datareader.read(args.datadir)
        voxelsize = metadata["voxelsize_mm"]
        obj = OrganDetection(data3d, voxelsize)

    else: # readydir
        obj = OrganDetection.fromDirectory(os.path.abspath(args.readydir))
        data3d = obj.getData3D()

    if args.dump is not None:
        for part in obj.masks_comp:
            try:
                obj.getPart(part, raw=True)
            except:
                print(traceback.format_exc())

        for part in obj.stats:
            try:
                obj.getStats(part, raw=True)
            except:
                print(traceback.format_exc())

        dumpdir = os.path.abspath(args.dump)
        if not os.path.exists(dumpdir): os.makedirs(dumpdir)
        obj.toDirectory(dumpdir)
        sys.exit(0)

    #########
    print("-----------------------------------------------------------")

    # body = obj.getBody()
    # fatlessbody = obj.getFatlessBody()
    # bones = obj.getBones()
    # lungs = obj.getLungs()
    #abdomen = obj.getAbdomen()
    vessels = obj.getVessels()
    aorta = obj.getAorta()
    venacava = obj.getVenaCava()

    # ed = sed3.sed3(data3d, contour=body); ed.show()
    # ed = sed3.sed3(data3d, contour=fatlessbody); ed.show()
    # ed = sed3.sed3(data3d, contour=bones); ed.show()
    # ed = sed3.sed3(data3d, contour=lungs); ed.show()
    # ed = sed3.sed3(data3d, contour=abdomen); ed.show()
    vc = np.zeros(vessels.shape, dtype=np.int8); vc[ vessels == 1 ] = 1
    vc[ aorta == 1 ] = 2; vc[ venacava == 1 ] = 3
    ed = sed3.sed3(data3d, contour=vc); ed.show()

    # bones_stats = obj.analyzeBones()
    # points_spine = bones_stats["spine"];  points_hip_joints = bones_stats["hip_joints"]

    # seeds = np.zeros(bones.shape)
    # for p in points_spine: seeds[p[0], p[1], p[2]] = 1
    # for p in points_hip_joints: seeds[p[0], p[1], p[2]] = 2
    # seeds = scipy.ndimage.morphology.grey_dilation(seeds, size=(1,5,5))
    # ed = sed3.sed3(data3d, contour=bones, seeds=seeds); ed.show()

    # vessels_stats = obj.analyzeVessels()
    # points_aorta = vessels_stats["aorta"];  points_vena_cava = vessels_stats["vena_cava"]

    # seeds = np.zeros(vessels.shape)
    # for p in points_aorta: seeds[p[0], p[1], p[2]] = 1
    # for p in points_vena_cava: seeds[p[0], p[1], p[2]] = 2
    # seeds = scipy.ndimage.morphology.grey_dilation(seeds, size=(1,5,5))
    # ed = sed3.sed3(data3d, contour=vessels, seeds=seeds); ed.show()







