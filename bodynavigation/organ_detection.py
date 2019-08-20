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

import sys, os
import copy
import json
import tempfile, shutil

import numpy as np
import scipy
import scipy.ndimage

import io3d

# run with: "python -m bodynavigation.organ_detection -h"
from .tools import NumpyEncoder, compressArray, decompressArray, toMemMap, delMemMap
from .organ_detection_algo import OrganDetectionAlgo
from .transformation import Transformation, TransformationNone
from .trainer3d import Trainer3D
from .files import getDefaultPAtlas, getDefaultClassifier

"""
Directly point to folder with patient ct data: (Must be in folder!)
/usr/bin/time -v python -m bodynavigation.organ_detection -d -i ./test_data/3Dircadb1.1
/usr/bin/time -v python -m bodynavigation.organ_detection -d -i ./test_data/sliver07-orig001
"""


class OrganDetection(object):
    """
    This class is only 'user interface', all algorithms are in OrganDetectionAlgo

    * getPart("bodypart") OR getBODYPART()
        - resizes output to corect shape (unless you use "raw=True")
        - saves (compressed) output in RAM for future calls
    """

    def __init__(
        self,
        data3d=None,
        voxelsize=[1, 1, 1],
        low_mem=True,
        clean_data=True,
        transformation_mode="spacing",
        crop=True,
        patlas_path=None,
        classifier_path=None,
    ):
        """
        * Values of input data should be in HU units (or relatively close). [air -1000, water 0]
            https://en.wikipedia.org/wiki/Hounsfield_scale
        * All coordinates and sizes are in [Z,Y,X] format
        * Expecting data3d to be corectly oriented
        * Voxel size is in mm

        low_mem - tries to lower memory usage by saving data3d and masks to temporary files
            on filesystem. Uses np.memmap which might not work with some functions.
        clean_data - if to run data3d through OrganDetectionAlgo.cleanData()
        transformation_mode - ["none","spacing","resize","registration"]
        crop - If transformation is alowed to crop/pad data
        """

        # empty undefined values
        self.data3d = np.zeros((1, 1, 1), dtype=np.int16)
        self.spacing = np.asarray(
            [1, 1, 1], dtype=np.float
        )  # internal self.data3d spacing
        self.spacing_source = np.asarray(
            [1, 1, 1], dtype=np.float
        )  # original data3d spacing
        self.transformation = TransformationNone(self.data3d.shape, self.spacing)
        self.registration_points_source = None  # source registration points
        self.registration_points_target = None  # target registration points
        # empty undefined patlas values
        self.patlas_path = None
        self.patlas_info = None
        self.patlas_transformation = None
        # empty undefined classifier values
        self.classifier_path = None
        self.classifier = {}  # keys are masks and values are Trainer3D

        # compressed masks - example: compression lowered memory usage to 0.042% for bones
        self.masks_comp = {
            "body": None,
            "fatlessbody": None,
            "lungs": None,
            "bones": None,
            "diaphragm": None,
            "vessels": None,
            "kidneys": None,
            "liver": None,
            "spleen": None,
        }

        # statistics and models
        self.stats = {}
        for part in self.masks_comp:
            self.stats[part] = None

        # create directory for temporary files
        self.low_mem = low_mem
        self.tempdir = tempfile.mkdtemp(prefix="organ_detection_")

        # fix transform mode string
        transformation_mode = transformation_mode.strip().lower()

        # init with data3d
        if data3d is not None:
            # remove noise and errors in data
            if clean_data:
                logger.info("Preparing input data...")
                data3d, body = OrganDetectionAlgo.cleanData(data3d, voxelsize)
            else:
                body = None

            # dump/read cleaned data3d from file
            if self.low_mem:
                data3d = toMemMap(
                    data3d, os.path.join(self.tempdir, "data3d_clean.dat")
                )

            # calculate transformation
            if transformation_mode == "none":
                self.transformation = TransformationNone(data3d.shape, voxelsize)
                self.setData3D(data3d, raw=False)  # set self.data3d
            else:
                # calc registration points
                logger.info("Preparing for calculation of registration points...")
                obj = OrganDetection(
                    data3d,
                    voxelsize,
                    low_mem=self.low_mem,
                    clean_data=False,
                    transformation_mode="none",
                    crop=False,
                )
                if body is not None:
                    obj.setPart("body", body, raw=False)
                obj._preloadParts(["body", "fatlessbody"])
                obj._preloadStats(["lungs", "bones"])
                self.registration_points_source = obj.getRegistrationPoints(
                    target=False
                )

                # init transformation from registration points
                logger.info("Init of transformation...")
                if transformation_mode == "spacing":
                    self.transformation = Transformation(
                        self.registration_points_source, resize=False, crop=crop
                    )
                elif transformation_mode == "resize":
                    self.transformation = Transformation(
                        self.registration_points_source, resize=True, crop=crop
                    )
                elif transformation_mode == "registration":
                    self.transformation = Transformation(
                        self.registration_points_source, registration=True
                    )
                else:
                    logger.error(
                        "Invalid 'transformation_mode'! '%s'" % str(transformation_mode)
                    )
                    sys.exit(2)

                # set self.data3d
                self.setData3D(data3d, raw=False)

                # recycle some processed masks
                logger.info("Recycling processed masks...")
                for part in ["body", "fatlessbody", "lungs"]:
                    self.setPart(part, obj.getPart(part, raw=False), raw=False)

                # cleanup
                del (obj, body)

            # remove dumped cleaned data3d
            if self.low_mem:
                delMemMap(data3d)

            # remember spacing
            self.spacing = self.transformation.getTargetSpacing()
            self.spacing_source = self.transformation.getSourceSpacing()

            # ed = sed3.sed3(self.data3d); ed.show()

            # load patlas and classifier
            if patlas_path is not None:
                self.loadPAtlas(patlas_path)
            if classifier_path is not None:
                self.loadClassifier(classifier_path)

    def __del__(self):
        """ Decontructor """

        # imports are set to None or deleted on app exit
        try:
            shutil.get_archive_formats()
        except:
            import shutil

        # remove tempdir
        shutil.rmtree(self.tempdir)

    @classmethod
    def fromReadyData(
        cls, data3d, data3d_info, masks={}, stats={}
    ):  # TODO - save and load custom patlas and classifier path
        """ For super fast testing """
        obj = cls()

        obj.transformation = Transformation.fromDict(data3d_info["transformation"])
        obj.registration_points_source = data3d_info["registration_points_source"]
        obj.registration_points_target = data3d_info["registration_points_target"]
        obj.spacing = np.asarray(data3d_info["spacing"], dtype=np.float)
        obj.spacing_source = obj.transformation.getSourceSpacing()
        obj.setData3D(data3d, raw=True)

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
        data3d, metadata = io3d.datareader.read(data3d_p, dataplus_format=False)
        with open(data3d_info_p, "r") as fp:
            data3d_info = json.load(fp)

        obj = cls()  # to get mask and stats names
        masks = {}
        stats = {}

        for part in obj.masks_comp:
            mask_p = os.path.join(path, "%s.dcm" % part)
            if os.path.exists(mask_p):
                tmp, _ = io3d.datareader.read(mask_p, dataplus_format=False)
                masks[part] = compressArray(tmp.astype(np.bool))
                del tmp

        for part in obj.stats:
            stats_p = os.path.join(path, "%s.json" % part)
            if os.path.exists(stats_p):
                with open(stats_p, "r") as fp:
                    tmp = json.load(fp)
                stats[part] = tmp

        return cls.fromReadyData(data3d, data3d_info, masks=masks, stats=stats)

    def toDirectory(self, path):
        """
        note: Masks look wierd when opened in ImageJ, but are saved correctly
        simpleitk 1.0.1 causes io3d.datawriter.write to hang, downgrade to 0.9.1
        """
        logger.info("Saving all processed data to directory: %s" % path)
        spacing = list(self.spacing)

        data3d_p = os.path.join(path, "data3d.dcm")
        io3d.datawriter.write(self.data3d, data3d_p, "dcm", {"voxelsize_mm": spacing})

        data3d_info_p = os.path.join(path, "data3d.json")
        data3d_info = {
            "spacing": copy.deepcopy(self.spacing),
            "transformation": copy.deepcopy(self.transformation.toDict()),
            "registration_points_source": copy.deepcopy(
                self.registration_points_source
            ),
            "registration_points_target": copy.deepcopy(
                self.registration_points_target
            ),
        }
        with open(data3d_info_p, "w") as fp:
            json.dump(data3d_info, fp, sort_keys=True, cls=NumpyEncoder)

        for part in self.masks_comp:
            if self.masks_comp[part] is None:
                continue
            mask_p = os.path.join(path, str("%s.dcm" % part))
            mask = self.getPart(part, raw=True).astype(np.int8)
            io3d.datawriter.write(mask, mask_p, "dcm", {"voxelsize_mm": spacing})
            del mask

        for part in self.stats:
            if self.stats[part] is None:
                continue
            stats_p = os.path.join(path, "%s.json" % part)
            with open(stats_p, "w") as fp:
                json.dump(
                    self.analyzePart(part, raw=True),
                    fp,
                    sort_keys=True,
                    cls=NumpyEncoder,
                )

    def toOutputCoordinates(self, vector):
        return np.asarray(self.transformation.transCoordinatesInv(vector))

    def getData3D(self, raw=False):
        if raw:
            return self.data3d.copy()
        else:
            return self.transformation.transDataInv(self.data3d, cval=-1024).copy()

    def setData3D(self, data3d, raw=False):
        if not raw:
            data3d = self.transformation.transData(data3d, cval=-1024)

        if self.low_mem:
            self.data3d = toMemMap(data3d, os.path.join(self.tempdir, "data3d.dat"))
        else:
            self.data3d = data3d

    ####################
    ### Segmentation ###
    ####################

    def getPart(self, part, raw=False):
        part = part.strip().lower()

        if part not in self.masks_comp:
            logger.error("Invalid bodypart '%s'! Returning empty mask!" % part)
            data = np.zeros(self.data3d.shape).astype(np.bool)

        elif self.masks_comp[part] is not None:
            data = decompressArray(self.masks_comp[part])

        else:
            if part == "body":
                data = OrganDetectionAlgo.getBody(self.data3d, self.spacing)
            elif part == "fatlessbody":
                self._preloadParts(["body"])
                data = OrganDetectionAlgo.getFatlessBody(
                    self.data3d, self.spacing, self.getBody(raw=True)
                )
            elif part == "lungs":
                self._preloadParts(["fatlessbody"])
                data = OrganDetectionAlgo.getLungs(
                    self.data3d, self.spacing, self.getFatlessBody(raw=True)
                )
            elif part == "bones":
                self._preloadParts(["fatlessbody", "lungs"])
                self._preloadStats(["lungs"])
                data = OrganDetectionAlgo.getBones(
                    self.data3d,
                    self.spacing,
                    self.getFatlessBody(raw=True),
                    self.getLungs(raw=True),
                    self.analyzeLungs(raw=True),
                )
            elif part == "diaphragm":
                self._preloadParts(["lungs", "body"])
                data = OrganDetectionAlgo.getDiaphragm(
                    self.data3d,
                    self.spacing,
                    self.getLungs(raw=True),
                    self.getBody(raw=True),
                )
            elif part == "vessels":
                self._preloadParts(["fatlessbody", "bones", "kidneys"])
                self._preloadStats(["bones"])
                data = OrganDetectionAlgo.getVessels(
                    self.data3d,
                    self.spacing,
                    self.getFatlessBody(raw=True),
                    self.getBones(raw=True),
                    self.analyzeBones(raw=True),
                    self.getKidneys(raw=True),
                )
            elif part == "kidneys":
                self._preloadParts(["fatlessbody", "diaphragm", "liver"])
                data = OrganDetectionAlgo.getKidneys(
                    self.data3d,
                    self.spacing,
                    self.getClassifierOutput(part, raw=True),
                    self.getFatlessBody(raw=True),
                    self.getDiaphragm(raw=True),
                    self.getLiver(raw=True),
                )
            elif part == "liver":
                self._preloadParts(["fatlessbody", "diaphragm"])
                data = OrganDetectionAlgo.getLiver(
                    self.data3d,
                    self.spacing,
                    self.getClassifierOutput(part, raw=True),
                    self.getFatlessBody(raw=True),
                    self.getDiaphragm(raw=True),
                )
            elif part == "spleen":
                self._preloadParts(["fatlessbody", "diaphragm"])
                data = OrganDetectionAlgo.getLiver(
                    self.data3d,
                    self.spacing,
                    self.getClassifierOutput(part, raw=True),
                    self.getFatlessBody(raw=True),
                    self.getDiaphragm(raw=True),
                )
            else:
                raise Exception(
                    "Looks like someone forgot to specify which algorithm to use for part '%s'..."
                    % part
                )

            self.masks_comp[part] = compressArray(data)

        if not raw:
            data = self.transformation.transDataInv(data, cval=0)
        return data

    def setPart(self, partname, data, raw=False):
        if not raw:
            data = self.transformation.transData(data, cval=0)
        if not np.all(self.data3d.shape == data.shape):
            logger.warning(
                "Manualy added segmented data does not have correct shape! %s != %s"
                % (str(self.data3d.shape), str(data.shape))
            )
        self.masks_comp[partname] = compressArray(data)

    def _preloadParts(self, partlist):
        """
        Lowers memory usage, by making sure that only data that are required for current
        process are loaded.
        """
        for part in partlist:
            if part not in self.masks_comp:
                logger.error("Invalid bodypart '%s'! Skipping preload!" % part)
                continue
            if self.masks_comp[part] is None:
                self.getPart(part, raw=True)

    def getBody(self, raw=False):
        return self.getPart("body", raw=raw)

    def getFatlessBody(self, raw=False):
        return self.getPart("fatlessbody", raw=raw)

    def getLungs(self, raw=False):
        return self.getPart("lungs", raw=raw)

    def getBones(self, raw=False):
        return self.getPart("bones", raw=raw)

    def getDiaphragm(self, raw=False):
        return self.getPart("diaphragm", raw=raw)

    def getVessels(self, raw=False):
        return self.getPart("vessels", raw=raw)

    def getKidneys(self, raw=False):
        return self.getPart("kidneys", raw=raw)

    def getLiver(self, raw=False):
        return self.getPart("liver", raw=raw)

    def getSpleen(self, raw=False):
        return self.getPart("spleen", raw=raw)

    #############################
    ### PAtlas and Classifier ###
    #############################

    def loadPAtlas(self, path=None):
        # load default patlas
        if path is None:
            logger.info("Using default PAtlas")
            path = os.path.join(self.tempdir, "patlas/")
            if not os.path.exists(path):
                os.makedirs(path)
            getDefaultPAtlas(path)

        # save path
        self.patlas_path = path

        # load patlas info
        with open(os.path.join(path, "PA_info.json"), "r") as fp:
            self.patlas_info = json.load(fp, encoding="utf-8")

        # init patlas transformation
        target_reg_points = self.getRegistrationPoints(target=True)
        patlas_reg_points = self.patlas_info["registration_points"]
        self.patlas_transformation = Transformation(
            patlas_reg_points, target_reg_points, registration=True
        )

    def getPartPAtlas(self, part, raw=False):
        if self.patlas_path is None:
            self.loadPAtlas()

        fpath = os.path.join(self.patlas_path, str("%s.dcm" % part))
        if not os.path.exists(fpath):
            logger.warning("Not found PAtlas mask path: %s" % fpath)
            data = np.zeros(self.data3d.shape, dtype=np.float)
        else:
            data, _ = io3d.datareader.read(fpath, dataplus_format=False)
            data = (
                data.astype(np.float32) / self.patlas_info["data_multiplication"]
            )  # convert back to percantage
            data = self.patlas_transformation.transData(data, cval=0)

        if not raw:
            data = self.transformation.transDataInv(data, cval=0)
        return data

    def loadClassifier(self, path=None):
        # load default classifier
        if path is None:
            logger.info("Using default Classifier")
            path = os.path.join(self.tempdir, "classifier/")
            if not os.path.exists(path):
                os.makedirs(path)
            getDefaultClassifier(path)

        # save path
        self.classifier_path = path

        # get list of availible classifiers
        part_classifiers = []
        for fname in [
            f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
        ]:
            fpath = os.path.join(path, fname)
            name, ext = os.path.splitext(fname)
            if ext == ".json":
                part_classifiers.append(name)

        # load classifier for parts
        for part in part_classifiers:
            fpath = os.path.join(self.classifier_path, str("%s.json" % part))
            self.classifier[part] = Trainer3D.fromFile(fpath)

    def getClassifier(self, part):
        if self.classifier_path is None:
            self.loadClassifier()

        if part not in self.classifier:
            logger.warning("Not found Classifier for part: %s" % part)

        return self.classifier[part]

    def getClassifierOutput(self, part, raw=False):
        ol = self.getClassifier(part)
        fv_kwargs = {}
        fv_kwargs["data3d"] = self.data3d
        fv_kwargs["patlas"] = self.getPartPAtlas(part, raw=True)
        # fv_kwargs["dist_fatlessbody_surface"] = self.distToPartSurface("fatlessbody", raw=True)
        # fv_kwargs["dist_diaphragm"] = self.distToPart("diaphragm", raw=True)
        data = ol.predict(self.data3d.shape, **fv_kwargs) != 0

        if not raw:
            data = self.transformation.transDataInv(data, cval=0)
        return data

    ########################
    ### Distance to Part ###
    ########################

    def _distToMask(self, mask, spacing):
        return scipy.ndimage.morphology.distance_transform_edt(
            mask == 0, sampling=spacing
        )

    def _distToMaskSurface(self, mask, spacing):
        """
        From inside of part to part surface or beyond.
        If object is touching border, that part is not counted as surface
        """
        return scipy.ndimage.morphology.distance_transform_edt(mask, sampling=spacing)

    def _distToMaskFull(self, mask, spacing):
        """ Distances under part surface are negative """
        data = self._distToMask(mask, spacing)
        data[data == 0] = (self._distToMaskSurface(mask, spacing) * (-1))[data == 0]
        return data

    def distToPart(self, part, raw=False):
        spacing = self.spacing if (raw is False) else self.spacing_source
        data = self._distToMask(self.getPart(part, raw=raw), spacing)
        return data

    def distToPartSurface(self, part, raw=False):
        spacing = self.spacing if (raw is False) else self.spacing_source
        data = self._distToMaskSurface(self.getPart(part, raw=raw), spacing)
        return data

    def distToPartFull(self, part, raw=False):
        """ Distances under part surface are negative """
        spacing = self.spacing if (raw is False) else self.spacing_source
        data = self._distToMaskFull(self.getPart(part, raw=raw), spacing)
        return data

    ##################
    ### Statistics ###
    ##################

    def getRegistrationPoints(self, target=False):
        if target is False and self.registration_points_source is not None:
            return self.registration_points_source
        elif target is True and self.registration_points_target is not None:
            return self.registration_points_target

        spacing = self.spacing if target else self.spacing_source
        reg_points = OrganDetectionAlgo.dataRegistrationPoints(
            spacing,
            self.getPart("body", raw=target),
            self.getPart("fatlessbody", raw=target),
            self.analyzePart("lungs", raw=target),
            self.analyzePart("bones", raw=target),
        )

        if target is False and self.registration_points_source is None:
            self.registration_points_source = reg_points
        elif target is True and self.registration_points_target is None:
            self.registration_points_target = reg_points

        return reg_points

    def analyzePart(self, part, raw=False):
        part = part.strip().lower()

        if part not in self.stats:
            logger.error(
                "Invalid stats bodypart '%s'! Returning empty dictionary!" % part
            )
            data = {}

        elif self.stats[part] is not None:
            data = copy.deepcopy(self.stats[part])

        else:
            if part == "lungs":
                self._preloadParts(["lungs", "fatlessbody"])
                data = OrganDetectionAlgo.analyzeLungs(
                    self.getLungs(raw=True),
                    self.spacing,
                    fatlessbody=self.getFatlessBody(raw=True),
                )
            if part == "bones":
                self._preloadParts(["fatlessbody", "bones"])
                self._preloadStats(["lungs"])
                data = OrganDetectionAlgo.analyzeBones(
                    self.getBones(raw=True),
                    self.spacing,
                    fatlessbody=self.getFatlessBody(raw=True),
                    lungs_stats=self.analyzeLungs(raw=True),
                )
            elif part == "vessels":
                self._preloadParts(["vessels", "bones", "liver"])
                self._preloadStats(["bones"])
                data = OrganDetectionAlgo.analyzeVessels(
                    data3d=self.data3d,
                    spacing=self.spacing,
                    vessels=self.getVessels(raw=True),
                    bones_stats=self.analyzeBones(raw=True),
                    liver=self.getLiver(raw=True),
                )

            self.stats[part] = copy.deepcopy(data)

        if not raw:
            if part == "lungs":
                data["start"] = self.toOutputCoordinates(
                    [
                        data["start"],
                        int(self.data3d.shape[1] / 2),
                        int(self.data3d.shape[2] / 2),
                    ]
                ).astype(np.int)[0]
                data["end"] = self.toOutputCoordinates(
                    [
                        data["end"],
                        int(self.data3d.shape[1] / 2),
                        int(self.data3d.shape[2] / 2),
                    ]
                ).astype(np.int)[0]
            if part == "bones":
                data["spine"] = [
                    tuple(self.toOutputCoordinates(p).astype(np.int))
                    for p in data["spine"]
                ]
                data["hips_joints"] = [
                    tuple(self.toOutputCoordinates(p).astype(np.int))
                    for p in data["hips_joints"]
                ]
                for i, p in enumerate(data["hips_start"]):
                    if p is None:
                        continue
                    data["hips_start"][i] = tuple(
                        self.toOutputCoordinates(p).astype(np.int)
                    )
            elif part == "vessels":
                data["aorta"] = [
                    tuple(self.toOutputCoordinates(p).astype(np.int))
                    for p in data["aorta"]
                ]
                data["vena_cava"] = [
                    tuple(self.toOutputCoordinates(p).astype(np.int))
                    for p in data["vena_cava"]
                ]
                data["liver"] = [
                    tuple(self.toOutputCoordinates(p).astype(np.int))
                    for p in data["liver"]
                ]
        return data

    def _preloadStats(self, statlist):
        """
        Lowers memory usage, by making sure that only data that are required for current
        process are loaded.
        """
        for part in statlist:
            if part not in self.stats:
                logger.error("Invalid stats bodypart '%s'! Skipping preload!" % part)
                continue
            if self.stats[part] is None:
                self.analyzePart(part, raw=True)

    def analyzeLungs(self, raw=False):
        return self.analyzePart("lungs", raw=raw)

    def analyzeBones(self, raw=False):
        return self.analyzePart("bones", raw=raw)

    def analyzeVessels(self, raw=False):
        return self.analyzePart("vessels", raw=raw)


if __name__ == "__main__":
    import argparse
    from .results_drawer import ResultsDrawer

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # input parser
    parser = argparse.ArgumentParser(description="Organ Detection")
    parser.add_argument("-i", "--datadir", default=None, help="path to data dir")
    parser.add_argument(
        "-r", "--readydir", default=None, help="path to ready data dir (for testing)"
    )
    parser.add_argument(
        "--dump", default=None, help="process and dump all data to path and exit"
    )
    parser.add_argument(
        "--draw",
        default=None,
        help='draw and show segmentation results for specified parts. example: "bones,vessels,lungs"',
    )
    parser.add_argument(
        "--drawdepth", action="store_true", help="draw image in solid depth mode."
    )
    parser.add_argument(
        "--show",
        default=None,
        help='Show one specific segmented part with sed3 viewer. example: "bones"',
    )
    parser.add_argument("--patlas", default=None, help="Path to custom patlas")
    parser.add_argument("--classifier", default=None, help="Path to custom classifier")
    parser.add_argument("-d", "--debug", action="store_true", help="run in debug mode")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    logging.getLogger("io3d").setLevel(logging.WARNING)

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
        print("Loading CT data...")
        # detect sole file or *.mhd
        datapath = args.datadir
        onlyfiles = sorted(
            [
                f
                for f in os.listdir(datapath)
                if os.path.isfile(os.path.join(datapath, f))
            ]
        )
        if len(onlyfiles) == 1:
            datapath = os.path.join(datapath, onlyfiles[0])
            print("Only one file datapath! Changing datapath to: ", datapath)
        else:
            for f in onlyfiles:
                if f.strip().lower().endswith(".mhd"):
                    datapath = os.path.join(datapath, f)
                    print("Detected *.mhd file! Changing datapath to: ", datapath)
                    break
        # lead data3d and init OrganDetection
        data3d, metadata = io3d.datareader.read(datapath, dataplus_format=False)
        voxelsize = metadata["voxelsize_mm"]
        obj = OrganDetection(data3d, voxelsize)
    else:  # readydir
        obj = OrganDetection.fromDirectory(os.path.abspath(args.readydir))
        voxelsize = obj.spacing_source
    # always display results on processed data3d
    data3d = obj.getData3D()
    # load custom patlas and classifier
    if args.patlas is not None:
        obj.loadPAtlas(args.patlas)
    if args.classifier is not None:
        obj.loadClassifier(args.classifier)

    if args.dump is not None:
        for part in obj.masks_comp:
            try:
                obj.getPart(part, raw=True)
            except:
                print(traceback.format_exc())

        for part in obj.stats:
            try:
                obj.analyzePart(part, raw=True)
            except:
                print(traceback.format_exc())

        dumpdir = os.path.abspath(args.dump)
        if not os.path.exists(dumpdir):
            os.makedirs(dumpdir)
        obj.toDirectory(dumpdir)
        sys.exit(0)

    if args.draw is not None:
        parts = [s.strip().lower() for s in args.draw.split(",")]
        masks = [obj.getPart(p) for p in parts]
        if args.drawdepth:
            rd = ResultsDrawer(mask_depth=True, default_volume_alpha=255)
        else:
            rd = ResultsDrawer()
        img = rd.drawImageAutocolor(data3d, voxelsize, volumes=masks)
        img.show()

    if args.show is not None:
        import sed3

        seg = obj.getPart(args.show)
        ed = sed3.sed3(data3d, contour=seg)
        ed.show()

    ####################################################################
    print("-----------------------------------------------------------")
    # data3d = obj.getData3D(raw=True)
    # lungs = obj.getLungs(raw=True)
    # bones = obj.getBones(raw=True)
    # vessels = obj.getVessels()
    # aorta = obj.getAorta()
    # venacava = obj.getVenaCava()

    # print(data3d.shape)
    # ed = sed3.sed3(data3d, contour=lungs); ed.show()
    # ed = sed3.sed3(data3d, contour=bones); ed.show()

    # bones_stats = obj.analyzeBones()
    # points_spine = bones_stats["spine"];  points_hips_joints = bones_stats["hips_joints"]
    # seeds = np.zeros(bones.shape)
    # for p in points_spine: seeds[p[0], p[1], p[2]] = 1
    # for p in points_hips_joints: seeds[p[0], p[1], p[2]] = 2
    # seeds = scipy.ndimage.morphology.grey_dilation(seeds, size=(1,5,5))
    # ed = sed3.sed3(data3d, contour=bones, seeds=seeds); ed.show()

    # vc = np.zeros(vessels.shape, dtype=np.int8); vc[ vessels == 1 ] = 1
    # vc[ aorta == 1 ] = 2; vc[ venacava == 1 ] = 3
    # ed = sed3.sed3(data3d, contour=vc); ed.show()

    # vessels_stats = obj.analyzeVessels()
    # points_aorta = vessels_stats["aorta"];  points_vena_cava = vessels_stats["vena_cava"]
    # points_liver = vessels_stats["liver"]
    # seeds = np.zeros(vessels.shape)
    # for p in points_aorta: seeds[p[0], p[1], p[2]] = 1
    # for p in points_vena_cava: seeds[p[0], p[1], p[2]] = 2
    # for p in points_liver: seeds[p[0], p[1], p[2]] = 3
    # seeds = scipy.ndimage.morphology.grey_dilation(seeds, size=(1,5,5))
    # ed = sed3.sed3(data3d, contour=vessels, seeds=seeds); ed.show()
