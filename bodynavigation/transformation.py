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
from .tools import resizeWithUpscaleNN, getDataPadding, cropArray, padArray
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

    def __init__(self, shape=None, spacing=None):
        super(TransformationNone, self).__init__()
        if shape is not None:
            self.source["shape"] = shape
            self.target["shape"] = shape
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

    # compare this data with output of OrganDetectionAlgo.dataRegistrationPoints()
    REGISTRATION_TARGET = {
        # this will make all following values in mm; DON'T CHANGE!!
        "spacing":np.asarray([1,1,1], dtype=np.float),
        "data_shape":(128,512,512), # not used
        "data_padding":[[0,0],[0,0],[0,0]], # not used
        "lungs_end":75,
        "hips_start":190,
        "fatlessbody_height":200,
        "fatlessbody_width":300,
        "fatlessbody_centroid":(0.5,0.5)
        }
    # offset of start and end of abdomen from lungs_end and hips_start; in mm
    REGISTRATION_ABDOMEN_OFFSET = 75

    def __init__(self, registration_points=None, resize=False, crop_z=False):
        """
        registration_points - output from OrganDetectionAlgo.dataRegistrationPoints()
        resize - if False only recalculates target spacing, If True resizes actual data.
        crop_z - crop on z-axis, so unly registred abdomen is in data
        """
        super(Transformation, self).__init__()

        # init some transformation variables
        self.trans["padding"] = [[0,0],[0,0],[0,0]]
        self.trans["cut_shape"] = (1,1,1)
        self.trans["coord_scale"] = np.asarray([1,1,1], dtype=np.float)
        self.trans["coord_intercept"] = np.asarray([0,0,0], dtype=np.float)

        # if missing input return undefined transformation
        if registration_points is None:
            return

        # define source variables
        self.source["spacing"] = np.asarray(registration_points["spacing"], dtype=np.float)
        self.source["shape"] = np.asarray(registration_points["shape"], dtype=np.int)

        # get registration parameters
        param = self._calcRegistrationParams(registration_points)

        # define crop/pad variables
        self.trans["padding"] = param["padding"]
        if not crop_z:
            self.trans["padding"][0] = [0,0]
        self.trans["cut_shape"] = np.asarray([
            registration_points["shape"][0]-np.sum(self.trans["padding"][0]),
            registration_points["shape"][1]-np.sum(self.trans["padding"][1]),
            registration_points["shape"][2]-np.sum(self.trans["padding"][2])
            ], dtype=np.int)

        # define scaling variables
        self.trans["reg_scale"] = np.asarray(param["scale"], dtype=np.float)
        self.trans["voxel_scale"] = np.asarray([1,1,1], dtype=np.float)
        self.trans["scale"] = np.asarray([1,1,1], dtype=np.float)
        if resize:
            self.target["spacing"] = self.REGISTRATION_TARGET["spacing"]
            self.trans["voxel_scale"] = np.asarray([
                self.source["spacing"][0] / self.target["spacing"][0],
                self.source["spacing"][1] / self.target["spacing"][1],
                self.source["spacing"][2] / self.target["spacing"][2]
                ], dtype=np.float)
            self.trans["scale"] = self.trans["reg_scale"]*self.trans["voxel_scale"]
        else:
            self.target["spacing"] = self.source["spacing"]*self.trans["reg_scale"]

        self.target["shape"] = np.asarray(np.round([
            self.trans["cut_shape"][0] * self.trans["scale"][0],
            self.trans["cut_shape"][1] * self.trans["scale"][1],
            self.trans["cut_shape"][2] * self.trans["scale"][2]
            ]), dtype=np.int)

        # for recalculating coordinates to output format ( vec*scale + intercept )
        self.trans["coord_scale"] = np.asarray([
            self.trans["cut_shape"][0] / self.target["shape"][0],
            self.trans["cut_shape"][1] / self.target["shape"][1],
            self.trans["cut_shape"][2] / self.target["shape"][2]
            ], dtype=np.float) # [z,y,x] - scale coords of cut and resized data
        self.trans["coord_intercept"] = np.asarray([
            self.trans["padding"][0][0],
            self.trans["padding"][1][0],
            self.trans["padding"][2][0]
            ], dtype=np.float) # [z,y,x] - move coords of just cut data

        # print debug
        logger.debug(self.toDict())

    def _calcRegistrationParams(self, reg_points):
        """
        How this works:
            1. get body padding
            2. recalc body y,x padding/crop based on desired centroid position (translation)
            3. calc body z padding/crop based on lungs_end, hips_start and desired offset
            4. calc data scaling

        How to use output:
            1. crop/pad array based on "padding" output
            2. scale spacing/data based on "scale"
        """
        ret = {}
        rp_s = copy.deepcopy(reg_points)
        rp_t = copy.deepcopy(self.REGISTRATION_TARGET)

        # get base data padding; negative needs to add padding, positive needs to crop
        ret["padding"] = copy.deepcopy(rp_s["padding"])

        # relative y,x centroid translation required
        centroid_trans = np.asarray(rp_t["fatlessbody_centroid"])-np.asarray(rp_s["fatlessbody_centroid"])
        # relative y,x centroid translation in voxels
        centroid_trans_voxel = [int(np.round(centroid_trans[0]*rp_s["shape"][1])), \
            int(np.round(centroid_trans[1]*rp_s["shape"][2]))]
        # calculate new data padding for y,x axis
        ret["padding"][1] = [ret["padding"][1][0]+centroid_trans_voxel[0], ret["padding"][1][1]]
        ret["padding"][2] = [ret["padding"][2][0]+centroid_trans_voxel[1], ret["padding"][2][1]]

        # calculate z-axis padding/offset/crop
        ret["padding"][0][0] = rp_s["lungs_end"]-int(self.REGISTRATION_ABDOMEN_OFFSET/rp_s["spacing"][0])
        ret["padding"][0][1] = (rp_s["shape"][0]-rp_s["hips_start"])-int(self.REGISTRATION_ABDOMEN_OFFSET/rp_s["spacing"][0])

        # calculate scale
        source_size_z = (rp_s["hips_start"]-rp_s["lungs_end"])*rp_s["spacing"][0]
        source_size_y = rp_s["fatlessbody_height"]*rp_s["spacing"][1]
        source_size_x = rp_s["fatlessbody_width"]*rp_s["spacing"][2]
        target_size_z = (rp_t["hips_start"]-rp_t["lungs_end"])*rp_t["spacing"][0]
        target_size_y = rp_t["fatlessbody_height"]*rp_t["spacing"][1]
        target_size_x = rp_t["fatlessbody_width"]*rp_t["spacing"][2]
        ret["scale"] = np.asarray([\
            target_size_z/source_size_z, target_size_y/source_size_y, target_size_x/source_size_x
            ], dtype=np.float)

        logger.debug(ret)
        return ret

    def transData(self, data3d, cval=0):
        data3d = cropArray(data3d, self.trans["padding"], padding_value=cval)
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

