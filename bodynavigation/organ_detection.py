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
import skimage.measure
import skimage.transform
import skimage.morphology
import skimage.segmentation
import skimage.feature

import io3d
import sed3

# run with: "python -m bodynavigation.organ_detection -h"
from .tools import getSphericalMask, binaryClosing, binaryFillHoles, \
    compressArray, decompressArray, getDataPadding, cropArray, padArray, polyfit3D, growRegion
from .organ_detection_algo import OrganDetectionAlgo

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
            "kidneys":None,
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
        body = OrganDetectionAlgo.getBody(data3d, voxelsize)
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
            fatlessbody = OrganDetectionAlgo.getFatlessBody(data3d, voxelsize, body); del(body)
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

    ################

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
                self._preloadParts(["lungs","fatlessbody"])
                data = OrganDetectionAlgo.getKidneys(self.data3d, self.spacing, self.getLungs(raw=True), \
                    self.getFatlessBody(raw=True))
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

        if not raw: data = self.toOutputFormat(data)
        return data

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
                self._preloadParts(["fatlessbody", "bones", "lungs"])
                data = OrganDetectionAlgo.analyzeBones( \
                data3d=self.data3d, spacing=self.spacing, fatlessbody=self.getFatlessBody(raw=True), \
                bones=self.getBones(raw=True), lungs=self.getLungs(raw=True) )
            elif part == "vessels":
                self._preloadParts(["vessels", "bones"])
                data = OrganDetectionAlgo.analyzeVessels( \
                data3d=self.data3d, spacing=self.spacing, vessels=self.getVessels(raw=True), \
                bones_stats=self.analyzeBones(raw=True) )

            self.stats[part] = copy.deepcopy(data)

        if not raw:
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







