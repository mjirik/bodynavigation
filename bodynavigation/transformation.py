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
        self.source = {}  # Source data3d info
        self.target = {}  # Target data3d info
        self.trans = {}  # Transformation variables

        # default 'empty' values
        self.source["spacing"] = np.asarray([1, 1, 1], dtype=np.float)
        self.source["shape"] = (1, 1, 1)
        self.target["spacing"] = np.asarray([1, 1, 1], dtype=np.float)
        self.target["shape"] = (1, 1, 1)

    def toDict(self):
        """ For saving transformation parameters to file """
        return {
            "source": copy.deepcopy(self.source),
            "target": copy.deepcopy(self.target),
            "trans": copy.deepcopy(self.trans),
        }

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
    """
    Normalization/Registration using rigid transformations
    """

    # compare this data with output of OrganDetectionAlgo.dataRegistrationPoints()
    DEFAULT_REGISTRATION_TARGET = {  # losely based on 3Dircadb1.1
        # this will make all following values in mm; DON'T CHANGE!!
        "spacing": np.asarray([1, 1, 1], dtype=np.float),  # used only in registration
        "shape": (265, 350, 400),  # used only in registration
        "padding": [[0, 0], [0, 0], [0, 0]],  # not used in target reg points
        "lungs_end": 75,
        "hips_start": 190,
        "fatlessbody_height": 200,
        "fatlessbody_width": 300,
        "fatlessbody_centroid": (0.5, 0.5),  # used only in registration
    }

    def __init__(
        self,
        reg_points_source=None,
        reg_points_target=None,
        resize=False,
        crop=True,
        registration=False,
    ):
        """
        reg_points_source - output from OrganDetectionAlgo.dataRegistrationPoints()
        reg_points_target - if None uses self.DEFAULT_REGISTRATION_TARGET
        resize - if False only recalculates target spacing, If True resizes actual data.
        crop - crop/pad array
        registration - if true does data registration (forces resize=True, crop=True)
        """
        super(Transformation, self).__init__()

        # init some transformation variables
        self.trans["padding"] = [[0, 0], [0, 0], [0, 0]]
        self.trans["cut_shape"] = (1, 1, 1)
        self.trans["coord_scale"] = np.asarray([1, 1, 1], dtype=np.float)
        self.trans["coord_intercept"] = np.asarray([0, 0, 0], dtype=np.float)

        # if missing input return undefined transformation
        if reg_points_source is None:
            return
        # if no transformation target, use defaults:
        if reg_points_target is None:
            reg_points_target = self.DEFAULT_REGISTRATION_TARGET
        # savekeep registration points
        self.trans["reg_points_source"] = copy.deepcopy(reg_points_source)
        self.trans["reg_points_target"] = copy.deepcopy(reg_points_target)

        # define source variables
        self.source["spacing"] = np.asarray(
            reg_points_source["spacing"], dtype=np.float
        )
        self.source["shape"] = np.asarray(reg_points_source["shape"], dtype=np.int)

        # get registration parameters
        param = self._calcRegistrationParams(reg_points_source, reg_points_target)

        # switch between normalization/registration
        if not registration:
            param["padding"] = reg_points_source["padding"]
        else:
            resize = True
            crop = True

        # crop/pad data
        if crop:
            self.trans["padding"] = param["padding"]
            self.trans["cut_shape"] = np.asarray(
                [
                    reg_points_source["shape"][0] - np.sum(self.trans["padding"][0]),
                    reg_points_source["shape"][1] - np.sum(self.trans["padding"][1]),
                    reg_points_source["shape"][2] - np.sum(self.trans["padding"][2]),
                ],
                dtype=np.int,
            )
        else:
            self.trans["padding"] = [[0, 0], [0, 0], [0, 0]]
            self.trans["cut_shape"] = reg_points_source["shape"]

        # resize voxels/data
        if resize:
            self.target["spacing"] = reg_points_target["spacing"]
            self.trans["scale"] = param["reg_scale"]
        else:
            self.target["spacing"] = param["spacing"]
            self.trans["scale"] = np.asarray([1, 1, 1], dtype=np.float)
        self.target["shape"] = np.round(
            self.trans["cut_shape"] * self.trans["scale"]
        ).astype(np.int)

        # for recalculating coordinates to output format ( vec*scale + intercept )
        self.trans["coord_scale"] = np.asarray(
            [
                self.trans["cut_shape"][0] / self.target["shape"][0],
                self.trans["cut_shape"][1] / self.target["shape"][1],
                self.trans["cut_shape"][2] / self.target["shape"][2],
            ],
            dtype=np.float,
        )  # [z,y,x] - scale coords of cut and resized data
        self.trans["coord_intercept"] = np.asarray(
            [
                self.trans["padding"][0][0],
                self.trans["padding"][1][0],
                self.trans["padding"][2][0],
            ],
            dtype=np.float,
        )  # [z,y,x] - move coords of just cut data

        # print debug
        logger.debug(self.toDict())

    def _calcRegistrationParams(self, reg_points_source, reg_points_target):
        """
        How to use output (normalization):
            1. replace source spacing with normalazed "spacing"

        How to use output (registration):
            1. crop/pad array based on "padding" output
            2. replace source spacing with normalized "spacing"
            3. scale data so that spacing is equal to rp_t["spacing"]
        """
        ret = {}
        rp_s = copy.deepcopy(reg_points_source)
        rp_t = copy.deepcopy(reg_points_target)

        ## normalization - calculate spacing scale
        source_size_z = (rp_s["hips_start"] - rp_s["lungs_end"]) * rp_s["spacing"][0]
        source_size_y = rp_s["fatlessbody_height"] * rp_s["spacing"][1]
        source_size_x = rp_s["fatlessbody_width"] * rp_s["spacing"][2]
        target_size_z = (rp_t["hips_start"] - rp_t["lungs_end"]) * rp_t["spacing"][0]
        target_size_y = rp_t["fatlessbody_height"] * rp_t["spacing"][1]
        target_size_x = rp_t["fatlessbody_width"] * rp_t["spacing"][2]
        ret["norm_scale"] = np.asarray(
            [
                target_size_z / source_size_z,
                target_size_y / source_size_y,
                target_size_x / source_size_x,
            ],
            dtype=np.float,
        )
        for i in range(3):
            if np.isinf(ret["norm_scale"][i]) or np.isnan(ret["norm_scale"][i]):
                logger.error("Scale at axis %i is inf/nan! Setting it to 1.0" % i)
                ret["norm_scale"][0] = 1.0
        # calc normalized spacing
        ret["spacing"] = np.asarray(rp_s["spacing"], dtype=np.float) * np.asarray(
            ret["norm_scale"], dtype=np.float
        )

        ## registration - padding/crop/offset/translation
        ret["reg_scale"] = np.asarray(ret["spacing"], dtype=np.float) / np.asarray(
            rp_t["spacing"], dtype=np.float
        )  # final data scaling
        reg_shape = np.round(
            np.asarray(rp_t["shape"], dtype=np.float) / ret["reg_scale"]
        ).astype(
            np.int
        )  # required shape before data scale
        ret["reg_scale"] = np.asarray(rp_t["shape"], dtype=np.float) / np.asarray(
            reg_shape, dtype=np.float
        )  # fix scale

        # calc required translation
        trans_yx_rel = np.asarray(rp_t["fatlessbody_centroid"]) - np.asarray(
            rp_s["fatlessbody_centroid"]
        )
        trans = np.asarray(
            np.round(
                np.asarray(
                    [
                        (
                            (rp_t["lungs_end"] * rp_t["spacing"][0])
                            - (rp_s["lungs_end"] * ret["spacing"][0])
                        )
                        / ret["spacing"][0],
                        trans_yx_rel[0] * rp_s["shape"][1],
                        trans_yx_rel[1] * rp_s["shape"][2],
                    ]
                )
            ),
            dtype=np.int,
        )

        # calc padding
        ret["padding"] = [[0, 0], [0, 0], [0, 0]]
        pads_sums = np.asarray(rp_s["shape"], dtype=np.int) - reg_shape
        # pad z
        ret["padding"][0][0] = -trans[0]
        ret["padding"][0][1] = pads_sums[0] - ret["padding"][0][0]
        # pad y,x
        ret["padding"][1][0] = (pads_sums[1] // 2) - trans[1]
        ret["padding"][1][1] = pads_sums[1] - ret["padding"][1][0]
        ret["padding"][2][0] = (pads_sums[2] // 2) - trans[2]
        ret["padding"][2][1] = pads_sums[2] - ret["padding"][2][0]

        logger.debug(ret)
        return ret

    def transData(self, data3d, cval=0):
        data3d = cropArray(data3d, self.trans["padding"], padding_value=cval)
        if not np.all(self.trans["cut_shape"] == self.target["shape"]):
            data3d = resizeWithUpscaleNN(data3d, np.asarray(self.target["shape"]))
        return data3d

    def transDataInv(self, data3d, cval=0):
        if not np.all(self.trans["cut_shape"] == self.target["shape"]):
            data3d = resizeWithUpscaleNN(data3d, np.asarray(self.trans["cut_shape"]))
        data3d = padArray(data3d, self.trans["padding"], padding_value=cval)
        return data3d

    def transCoordinates(self, coords):
        return (
            np.asarray(coords) - np.asarray(self.trans["coord_intercept"])
        ) / np.asarray(self.trans["coord_scale"])

    def transCoordinatesInv(self, coords):
        return (
            np.asarray(coords) * np.asarray(self.trans["coord_scale"])
        ) + np.asarray(self.trans["coord_intercept"])
