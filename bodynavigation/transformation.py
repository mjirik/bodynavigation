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

import copy

import numpy as np
import skimage.measure

# run with: "python -m bodynavigation.organ_detection -h"
from .tools import getSphericalMask, binaryClosing, binaryFillHoles, resizeWithUpscaleNN, \
    compressArray, decompressArray, getDataPadding, cropArray, padArray, polyfit3D, growRegion
from .organ_detection_algo import OrganDetectionAlgo

class TransformationInf(object):
    """ Parent class for other Transformation subclasses. ('interface' class) """

    def __init__(self):
        self.source = {} # Source data3d info
        self.target = {} # Target data3d info
        self.trans = {} # Transformation variables

        # default 'empty' values
        self.source["spacing"] = np.asarray([1,1,1], dtype=np.float)
        self.source["shape"] = (1,1,1)
        self.target["spacing"] = np.asarray([1,1,1], dtype=np.float)
        self.target["shape"] = (1,1,1)

    def toDict(self):
        """ For saving transformation parameters to file """
        return {"source": copy.deepcopy(self.source), "target": copy.deepcopy(self.target), \
            "trans": copy.deepcopy(self.trans)}

    @classmethod
    def fromDict(cls, data_dict):
        """ For loading transformation parameters from file """
        obj = cls()
        obj.source = copy.deepcopy(data_dict["source"])
        obj.target = copy.deepcopy(data_dict["target"])
        obj.trans = copy.deepcopy(data_dict["trans"])
        return obj

    # Getters

    def getSourceSpacing(self):
        return self.source["spacing"]

    def getTargetSpacing(self):
        return self.target["spacing"]

    def getSourceShape(self):
        return self.source["shape"]

    def getTargetShape(self):
        return self.target["shape"]

    # Functions that need to be implemented in child classes

    def transData(self, data3d, cval=0):
        """
        Transformation of numpy array
        cval - fill value for data outside the space of untransformed data
        """
        raise NotImplementedError

    def transDataInv(self, data3d, cval=0):
        """
        Inverse transformation of numpy array
        cval - fill value for data outside the space of untransformed data
        """
        raise NotImplementedError

    def transCoordinates(self, coords):
        """ Transformation of coordinates (list of lists) """
        raise NotImplementedError

    def transCoordinatesInv(self, coords):
        """ Inverse transformation of coordinates (list of lists) """
        raise NotImplementedError


class TransformationNone(TransformationInf):
    """
    Transformation that returns unchanged input.
    Useful in __init__ functions as default value.
    """

    def __init__(self, data3d=None, spacing=None):
        super(TransformationNone, self).__init__()
        if data3d is not None:
            self.source["shape"] = data3d.shape
            self.target["shape"] = data3d.shape
        if spacing is not None:
            self.source["spacing"] = np.asarray(spacing, dtype=np.float)
            self.target["spacing"] = np.asarray(spacing, dtype=np.float)

    def transData(self, data3d, cval=0):
        return data3d

    def transDataInv(self, data3d, cval=0):
        return data3d

    def transCoordinates(self, coords):
        return coords

    def transCoordinatesInv(self, coords):
        return coords


class Transformation(TransformationInf):

    # normalized size of fatless body on [Y,X] in mm
    NORMED_FATLESS_BODY_SIZE = [200,300]
    # [186.9, 247.4]mm - 3Dircadb1.1
    # [180.6, 256.4]mm - 3Dircadb1.2
    # [139.2, 253.1]mm - 3Dircadb1.19
    # [157.2, 298.8]mm - imaging.nci.nih.gov_NSCLC-Radiogenomics-Demo_1.3.6.1.4.1.14519.5.2.1.4334.1501.332134560740628899985464129848
    # [192.2, 321.4]mm - imaging.nci.nih.gov_NSCLC-Radiogenomics-Demo_1.3.6.1.4.1.14519.5.2.1.4334.1501.117528645891554472837507616577
    # [153.3, 276.3]mm - imaging.nci.nih.gov_NSCLC-Radiogenomics-Demo_1.3.6.1.4.1.14519.5.2.1.4334.1501.164951610473301901732855875499
    # [190.4, 304.6]mm - imaging.nci.nih.gov_NSCLC-Radiogenomics-Demo_1.3.6.1.4.1.14519.5.2.1.4334.1501.252450948398764180723210762820

    # calculating median widths and sizes from slices in lungs_end:lungs_end+THIS, in mm
    NORMED_FATLESS_BODY_SIZE_Z_RANGE = 200

    def __init__(self, data3d=None, spacing=None, norm_scale_enable=True, voxel_scale_enable=True, \
        z_scale_enable=True, body=None):
        """
        data3d, spacing - 3d CT data and voxelsize
        norm_scale_enable - try to normalize width and height of body (default True)
        voxel_scale_enable - scale voxels to 1x1x1mm (default True)
        z_scale_enable - enables scaling of data on z-axis, roughly 2x memory size.
            If disabled, target spacing will be changed so that no rescaling will happen \
            on z-axis when calling transData(). (default False)
        """
        super(Transformation, self).__init__()

        # init transformation variables
        self.trans["cut_shape"] = (0,0,0)
        self.trans["padding"] = [[0,0],[0,0],[0,0]]
        self.trans["coord_scale"] = np.asarray([1,1,1], dtype=np.float)
        self.trans["coord_intercept"] = np.asarray([0,0,0], dtype=np.float)

        # if missing input return undefined transformation
        if data3d is None or spacing is None:
            return

        # save source data3d info
        self.source["shape"] = data3d.shape
        self.source["spacing"] = spacing

        # Detect data padding and crop work data3d
        logger.debug("Detecting array padding")
        if body is None: body = OrganDetectionAlgo.getBody(data3d, spacing)
        padding = getDataPadding(body)
        data3d = cropArray(data3d, padding)
        body = cropArray(body, padding)
        self.trans["padding"] = padding
        self.trans["cut_shape"] = body.shape

        # Get NORMALIZATION SCALE
        self.trans["norm_scale"] = np.asarray([1,1,1], dtype=np.float)
        if norm_scale_enable:
            # Fatless body segmentation
            fatlessbody = OrganDetectionAlgo.getFatlessBody(data3d, spacing, body)
            del(body)

            # get lungs end # TODO - this kills RAM on full body shots
            logger.debug("Detecting end of lungs")
            lungs = OrganDetectionAlgo.getLungs(data3d, spacing, fatlessbody)
            if np.sum(lungs) == 0:
                lungs_end = 0
            else:
                lungs_end = data3d.shape[0] - getDataPadding(lungs)[0][1]
            del(lungs)

            logger.debug("Calculating size normalization scale")
            # get median body widths and heights
            # from just [lungs_end:(lungs_end+self.NORMED_FATLESS_BODY_SIZE_Z_RANGE/spacing[0]),:,:]
            widths = []; heights = []
            for z in range(lungs_end, min(int(lungs_end+self.NORMED_FATLESS_BODY_SIZE_Z_RANGE/spacing[0]), fatlessbody.shape[0])):
                if np.sum(fatlessbody[z,:,:]) == 0: continue
                spads = getDataPadding(fatlessbody[z,:,:])
                heights.append( fatlessbody[z,:,:].shape[0]-np.sum(spads[0]) )
                widths.append( fatlessbody[z,:,:].shape[1]-np.sum(spads[1]) )

            if len(widths) != 0:
                size_v = [ np.median(heights), np.median(widths) ]
            else:
                logger.warning("Could not detect median body (abdomen) width and height! Using size of middle slice for normalization.")
                size_v = []
                for dim, pad in enumerate(getDataPadding(fatlessbody[int(fatlessbody.shape[0]/2),:,:])):
                    size_v.append( fatlessbody.shape[dim+1]-np.sum(pad) )
            del(fatlessbody)

            # Calculate NORMALIZATION SCALE
            size_mm = [ size_v[0]*spacing[1], size_v[1]*spacing[2] ] # fatlessbody size in mm on X and Y axis
            norm_scale = [ None, self.NORMED_FATLESS_BODY_SIZE[0]/size_mm[0], self.NORMED_FATLESS_BODY_SIZE[1]/size_mm[1] ]
            norm_scale[0] = (norm_scale[1]+norm_scale[2])/2 # scaling on z-axis is average of scaling on x,y-axis
            self.trans["norm_scale"] = norm_scale # not used outside of init
        else:
            del(body)

        # Get VOXEL SCALE, scaling so that data has voxelsize 1x1x1mm
        self.trans["voxel_scale"] = np.asarray([1,1,1], dtype=np.float)
        if voxel_scale_enable:
            logger.debug("Calculating voxelsize 1x1x1mm scale")
            self.target["spacing"] = np.asarray([1,1,1], dtype=np.float)
            voxel_scale = np.asarray([ \
                self.source["spacing"][0] / self.target["spacing"][0], \
                self.source["spacing"][1] / self.target["spacing"][1], \
                self.source["spacing"][2] / self.target["spacing"][2] \
                ], dtype=np.float)
            self.trans["voxel_scale"] = voxel_scale # not used outside of init

        # Get FINAL SCALE, scaling so that final data is normalized and has voxelsize 1x1x1/??x1x1mm
        self.trans["scale"] = self.trans["norm_scale"]*self.trans["voxel_scale"]
        if not z_scale_enable:
            self.target["spacing"][0] = self.source["spacing"][0] * self.trans["norm_scale"][0] # TODO - test if is correct
            self.trans["scale"][0] = 1.0
        logger.debug("Final scale: %s" % str(self.trans["scale"]))

        # Calculate transformed shape and spacing
        logger.debug("Calculating transformed shape and help variables")
        self.target["shape"] = np.asarray(np.round([ \
            self.trans["cut_shape"][0] * self.trans["scale"][0], \
            self.trans["cut_shape"][1] * self.trans["scale"][1], \
            self.trans["cut_shape"][2] * self.trans["scale"][2] \
            ]), dtype=np.int)

        # for recalculating coordinates to output format ( vec*scale + intercept )
        self.trans["coord_scale"] = np.asarray([ \
            self.trans["cut_shape"][0] / self.target["shape"][0], \
            self.trans["cut_shape"][1] / self.target["shape"][1], \
            self.trans["cut_shape"][2] / self.target["shape"][2] ], dtype=np.float) # [z,y,x] - scale coords of cut and resized data
        self.trans["coord_intercept"] = np.asarray([ \
            self.trans["padding"][0][0], \
            self.trans["padding"][1][0], \
            self.trans["padding"][2][0] ], dtype=np.float) # [z,y,x] - move coords of just cut data

    def transData(self, data3d, cval=0):
        data3d = cropArray(data3d, self.trans["padding"])
        data3d = resizeWithUpscaleNN(data3d, np.asarray(self.target["shape"]))
        return data3d

    def transDataInv(self, data3d, cval=0):
        data3d = resizeWithUpscaleNN(data3d, np.asarray(self.trans["cut_shape"]))
        data3d = padArray(data3d, self.trans["padding"], padding_value=cval)
        return data3d

    def transCoordinates(self, coords):
        return ( np.asarray(coords) - np.asarray(self.trans["coord_intercept"]) ) / np.asarray(self.trans["coord_scale"])

    def transCoordinatesInv(self, coords):
        return ( np.asarray(coords) * np.asarray(self.trans["coord_scale"]) ) + np.asarray(self.trans["coord_intercept"])

