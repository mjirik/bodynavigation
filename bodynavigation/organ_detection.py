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
import copy
import json

import numpy as np
import scipy
import scipy.ndimage

import io3d
import sed3

# run with: "python -m bodynavigation.organ_detection -h"
from .tools import NumpyEncoder, compressArray, decompressArray
from .organ_detection_algo import OrganDetectionAlgo
from .transformation import Transformation, TransformationNone

class OrganDetection(object):
    """
    This class is only 'user interface', all algorithms are in OrganDetectionAlgo

    * getORGAN()
        - resizes output to corect shape (unless you use "raw=True")
        - saves (compressed) output in RAM for future calls
    """

    def __init__(self, data3d=None, voxelsize=[1,1,1]):
        """
        * Values of input data should be in HU units (or relatively close). [air -1000, water 0]
            https://en.wikipedia.org/wiki/Hounsfield_scale
        * All coordinates and sizes are in [Z,Y,X] format
        * Expecting data3d to be corectly oriented
        * Voxel size is in mm
        """

        # empty undefined values
        self.data3d = np.zeros((1,1,1), dtype=np.int16)
        self.spacing = np.asarray([1,1,1], dtype=np.float) # internal self.data3d spacing
        self.spacing_source = np.asarray([1,1,1], dtype=np.float) # original data3d spacing
        self.transformation = TransformationNone(self.data3d, self.spacing)

        # compressed masks - example: compression lowered memory usage to 0.042% for bones
        self.masks_comp = {
            "body":None,
            "fatlessbody":None,
            "lungs":None,
            "diaphragm":None,
            "kidneys":None,
            "bones":None,
            "abdomen":None,
            "vessels":None,
            "aorta":None,
            "venacava":None,
            }

        # statistics and models
        self.stats = {
            "lungs":None,
            "bones":None,
            "vessels":None
            }

        # init with data3d
        if data3d is not None:
            logger.info("Preparing input data...")

            # remove noise and errors in data
            data3d, body = OrganDetectionAlgo.cleanData(data3d, voxelsize)

            # Data Registration and Transformation
            self.transformation = Transformation(data3d, voxelsize, body=body)
            self.data3d = self.transformation.transData(data3d)
            #ed = sed3.sed3(self.data3d); ed.show()

            # remember spacing
            self.spacing = self.transformation.getTargetSpacing()
            self.spacing_source = self.transformation.getSourceSpacing()

    @classmethod
    def fromReadyData(cls, data3d, data3d_info, masks={}, stats={}):
        """ For super fast testing """
        obj = cls()

        obj.data3d = data3d
        obj.transformation = Transformation.fromDict(data3d_info["transformation"])
        obj.spacing = np.asarray(data3d_info["spacing"], dtype=np.float)
        obj.spacing_source = obj.transformation.getSourceSpacing()


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
        with open(data3d_info_p, 'r') as fp:
            data3d_info = json.load(fp)

        obj = cls() # to get mask and stats names
        masks = {}; stats = {}

        for part in obj.masks_comp:
            mask_p = os.path.join(path, "%s.dcm" % part)
            if os.path.exists(mask_p):
                tmp, _ = io3d.datareader.read(mask_p, dataplus_format=False)
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
        """
        note: Masks look wierd when opened in ImageJ, but are saved correctly
        simpleitk 1.0.1 causes io3d.datawriter.write to hang, downgrade to 0.9.1
        """
        logger.info("Saving all processed data to directory: %s" % path)
        spacing = list(self.spacing)

        data3d_p = os.path.join(path, "data3d.dcm")
        io3d.datawriter.write(self.data3d, data3d_p, 'dcm', {'voxelsize_mm': spacing})

        data3d_info_p = os.path.join(path, "data3d.json")
        data3d_info = {
            "spacing":copy.deepcopy(self.spacing),
            "transformation": copy.deepcopy(self.transformation.toDict())
            }
        with open(data3d_info_p, 'w') as fp:
            json.dump(data3d_info, fp, sort_keys=True, cls=NumpyEncoder)


        for part in self.masks_comp:
            if self.masks_comp[part] is None: continue
            mask_p = os.path.join(path, str("%s.dcm" % part))
            mask = self.getPart(part, raw=True).astype(np.int8)
            io3d.datawriter.write(mask, mask_p, 'dcm', {'voxelsize_mm': spacing})
            del(mask)

        for part in self.stats:
            if self.stats[part] is None: continue
            stats_p = os.path.join(path, "%s.json" % part)
            with open(stats_p, 'w') as fp:
                json.dump(self.getStats(part, raw=True), fp, sort_keys=True, cls=NumpyEncoder)

    def toOutputCoordinates(self, vector):
        return self.transformation.transCoordinatesInv(vector)

    def getData3D(self, raw=False):
        if raw:
            return self.data3d.copy()
        else:
            self.transformation.transDataInv(self.data3d.copy(), cval=-1024)

    ####################
    ### Segmentation ###
    ####################

    def getPart(self, part, raw=False):
        part = part.strip().lower()

        if part not in self.masks_comp:
            logger.error("Invalid bodypart '%s'! Returning empty mask!" % part)
            data = np.zeros(self.data3d.shape)

        elif self.masks_comp[part] is not None:
            data = decompressArray(self.masks_comp[part])

        else:
            if part == "body":
                data = OrganDetectionAlgo.getBody(self.data3d, self.spacing)
            elif part == "fatlessbody":
                self._preloadParts(["body",])
                data = OrganDetectionAlgo.getFatlessBody(self.data3d, self.spacing, self.getBody(raw=True))
            elif part == "lungs":
                self._preloadParts(["fatlessbody",])
                data = OrganDetectionAlgo.getLungs(self.data3d, self.spacing, self.getFatlessBody(raw=True))
            elif part == "diaphragm":
                self._preloadParts(["lungs",])
                data = OrganDetectionAlgo.getDiaphragm(self.data3d, self.spacing, self.getLungs(raw=True))
            elif part == "kidneys":
                self._preloadParts(["fatlessbody",])
                data = OrganDetectionAlgo.getKidneys(self.data3d, self.spacing, self.getFatlessBody(raw=True), \
                    self.analyzeLungs(raw=True) )
            elif part == "bones":
                self._preloadParts(["fatlessbody","lungs", "kidneys"])
                data = OrganDetectionAlgo.getBones(self.data3d, self.spacing, self.getFatlessBody(raw=True), \
                    self.getLungs(raw=True), self.getKidneys(raw=True) )
            elif part == "abdomen":
                self._preloadParts(["fatlessbody","diaphragm"])
                data = OrganDetectionAlgo.getAbdomen(self.data3d, self.spacing, self.getFatlessBody(raw=True), \
                    self.getDiaphragm(raw=True), self.analyzeBones(raw=True))
            elif part == "vessels":
                self._preloadParts(["bones",])
                data = OrganDetectionAlgo.getVessels(self.data3d, self.spacing, \
                    self.getBones(raw=True), self.analyzeBones(raw=True) )
            elif part == "aorta":
                self._preloadParts(["vessels",])
                data = OrganDetectionAlgo.getAorta(self.data3d, self.spacing, self.getVessels(raw=True), \
                    self.analyzeVessels(raw=True) )
            elif part == "venacava":
                self._preloadParts(["vessels",])
                data = OrganDetectionAlgo.getVenaCava(self.data3d, self.spacing, self.getVessels(raw=True), \
                    self.analyzeVessels(raw=True) )

            self.masks_comp[part] = compressArray(data)

        if not raw:
            data = self.transformation.transDataInv(data, cval=0)
        return data

    def setPart(self, partname, data, raw=False): # TODO - test this, use this (in patlas?)
        if not raw:
            data = self.transformation.transData(data, cval=0)
        if not np.all(self.data3d.shape == data.shape):
            logger.warning("Manualy added segmented data does not have correct shape! %s != %s" % (str(self.data3d.shape), str(data.shape)))
        self.masks_comp[partname] = compressArray(data)

    def _preloadParts(self, partlist):
        """ Lowers memory usage """
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

    def getDiaphragm(self, raw=False):
        return self.getPart("diaphragm", raw=raw)

    def getKidneys(self, raw=False):
        return self.getPart("kidneys", raw=raw)

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

    ##################
    ### Statistics ###
    ##################

    def getStats(self, part, raw=False):
        part = part.strip().lower()

        if part not in self.stats:
            logger.error("Invalid stats bodypart '%s'! Returning empty dictionary!" % part)
            data = {}

        elif self.stats[part] is not None:
            data = copy.deepcopy(self.stats[part])

        else:
            if part == "lungs":
                self._preloadParts(["lungs",])
                data =  OrganDetectionAlgo.analyzeLungs(self.data3d, self.spacing, \
                    lungs=self.getLungs(raw=True))
            if part == "bones":
                self._preloadParts(["fatlessbody", "bones"])
                data = OrganDetectionAlgo.analyzeBones( \
                data3d=self.data3d, spacing=self.spacing, fatlessbody=self.getFatlessBody(raw=True), \
                bones=self.getBones(raw=True), lungs_stats=self.analyzeLungs(raw=True) )
            elif part == "vessels":
                self._preloadParts(["vessels", "bones"])
                data = OrganDetectionAlgo.analyzeVessels( \
                data3d=self.data3d, spacing=self.spacing, vessels=self.getVessels(raw=True), \
                bones_stats=self.analyzeBones(raw=True) )

            self.stats[part] = copy.deepcopy(data)

        if not raw:
            if part == "lungs":
                data["start"] = self.toOutputCoordinates( \
                    [data["start"], int(self.data3d.shape[1]/2), int(self.data3d.shape[2]/2)] \
                    ).astype(np.int)[0]
                data["end"] = self.toOutputCoordinates( \
                    [data["end"], int(self.data3d.shape[1]/2), int(self.data3d.shape[2]/2)] \
                    ).astype(np.int)[0]
            if part == "bones":
                data["spine"] = [ tuple(self.toOutputCoordinates(p).astype(np.int)) for p in data["spine"] ]
                data["hip_joints"] = [ tuple(self.toOutputCoordinates(p).astype(np.int)) for p in data["hip_joints"] ]
                for i, p in enumerate(data["hip_start"]):
                    if p is None: continue
                    data["hip_start"][i] = tuple(self.toOutputCoordinates(p).astype(np.int))
            elif part == "vessels":
                data["aorta"] = [ tuple(self.toOutputCoordinates(p).astype(np.int)) for p in data["aorta"] ]
                data["vena_cava"] = [ tuple(self.toOutputCoordinates(p).astype(np.int)) for p in data["vena_cava"] ]
        return data

    def analyzeLungs(self, raw=False):
        return self.getStats("lungs", raw=raw)

    def analyzeBones(self, raw=False):
        return self.getStats("bones", raw=raw)

    def analyzeVessels(self, raw=False):
        return self.getStats("vessels", raw=raw)



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
        data3d, metadata = io3d.datareader.read(args.datadir, dataplus_format=False)
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
    kidneys = obj.getKidneys()
    # abdomen = obj.getAbdomen()
    # vessels = obj.getVessels()
    # aorta = obj.getAorta()
    # venacava = obj.getVenaCava()

    # ed = sed3.sed3(data3d, contour=body); ed.show()
    # ed = sed3.sed3(data3d, contour=fatlessbody); ed.show()
    # ed = sed3.sed3(data3d, contour=bones); ed.show()
    # ed = sed3.sed3(data3d, contour=lungs); ed.show()
    ed = sed3.sed3(data3d, contour=kidneys); ed.show()
    # ed = sed3.sed3(data3d, contour=abdomen); ed.show()
    # vc = np.zeros(vessels.shape, dtype=np.int8); vc[ vessels == 1 ] = 1
    # vc[ aorta == 1 ] = 2; vc[ venacava == 1 ] = 3
    # ed = sed3.sed3(data3d, contour=vc); ed.show()

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







