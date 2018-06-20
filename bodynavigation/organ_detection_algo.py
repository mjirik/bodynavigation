#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
from __future__ import print_function   # print("text")
from __future__ import division         # 2/3 == 0.666; 2//3 == 0
from __future__ import absolute_import  # 'import submodule2' turns into 'from . import submodule2'
from builtins import range              # replaces range with xrange

import logging
logger = logging.getLogger(__name__)

import sys, os
from operator import itemgetter
from itertools import groupby

import numpy as np
import scipy
import scipy.ndimage
import skimage.measure
import skimage.transform
import skimage.morphology
import skimage.segmentation
import skimage.feature

import sed3

# run with: "python -m bodynavigation.organ_detection -h"
from .tools import getSphericalMask, binaryClosing, binaryFillHoles, getDataPadding, \
    cropArray, padArray, polyfit3D, growRegion

class OrganDetectionAlgo(object):

    BODY_THRESHOLD = -300

    KIDNEYS_THRESHOLD_1 = 180 #150 # 180
    KIDNEYS_THRESHOLD_2 = 180 #250 # 180
    KIDNEYS_MIN_VOLUME = 100000 # experimantal volume of kidneys is about 200k mm3

    VESSELS_THRESHOLD = 110 # 145
    VESSELS_SPINE_WIDTH = 22 # from center
    VESSELS_SPINE_HEIGHT = 30 # from center

    VESSELS_AORTA_RADIUS = 12
    VESSELS_VENACAVA_RADIUS = 12

    @classmethod
    def cleanData(cls, data3d, spacing, body=None):
        """
        Filters out noise, removes some errors in data, sets undefined voxel value to -1024, etc ...
        """
        # fix for io3d <-512;511> value range bug, that is caused by hardcoded slope 0.5 in dcmreader
        if np.min(data3d) >= -512:
            logger.debug("Fixing io3d <-512;511> value range bug")
            data3d = data3d * 2

        # set padding value to -1024 (undefined voxel values in space outside of senzor range)
        logger.debug("Setting 'padding' value")
        data3d[ data3d == data3d[0,0,0] ] = -1024

        # limit value range to <-1024;32000> so it can fit into int16
        # [ data3d < -1024 ] => less dense then air - padding values
        # [ data3d > 32000  ] => near limit of int16
        logger.debug("Converting to int16")
        data3d[ data3d < -1024 ] = -1024
        data3d[ data3d > 32000 ] = 32000
        data3d = data3d.astype(np.int16)

        # filter out noise - median filter with radius 1 (kernel 3x3x3)
        logger.debug("Removing noise with filter")
        data3d = scipy.ndimage.filters.median_filter(data3d, 3)
        # ed = sed3.sed3(data3d); ed.show()

        # remove high brightness errors near edges of valid data (takes about 70s)
        logger.debug("Removing high brightness errors near edges of valid data") # TODO - clean this part up
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
            # lower all values near border bigger then BODY_THRESHOLD closer to BODY_THRESHOLD
            dst = scipy.ndimage.morphology.distance_transform_edt(tmp).astype(np.float)
            dst = dst/np.max(dst)
            dst[ dst != 0 ] = 0.01**dst[ dst != 0 ]; dst[ dst == 0 ] = 1.0

            mask = data3d[z,:,:] > cls.BODY_THRESHOLD
            data3d[z,:,:][mask] = ( \
                ((data3d[z,:,:][mask].astype(np.float)+300)*dst[mask])-300 \
                ).astype(np.int16) # TODO - use cls.BODY_THRESHOLD
        del(valid_mask, dst)
        # ed = sed3.sed3(data3d); ed.show()

        # remove anything that is not in body volume
        logger.debug("Removing all data outside of segmented body")
        if body is None:
            body = cls.getBody(data3d, spacing)
        data3d[ body == 0 ] = -1024

        #ed = sed3.sed3(data3d); ed.show()
        return data3d, body

    ####################
    ### Segmentation ###
    ####################

    @classmethod
    def getBody(cls, data3d, spacing):
        """
        Input: noiseless data3d
        Returns binary mask representing body volume (including most cavities)
        """
        logger.info("getBody()")
        # segmentation of body volume
        body = (data3d > cls.BODY_THRESHOLD).astype(np.bool)

        # fill holes
        body = binaryFillHoles(body, z_axis=True)

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
    def getFatlessBody(cls, data3d, spacing, body): # TODO - ignore nipples when creating convex hull
        """
        Returns convex hull of body without fat and skin
        """
        logger.info("getFatlessBody()")
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
    def getLungs(cls, data3d, spacing, fatlessbody): # TODO - check how much memory this eats; lower memory usage
        """ Expects lungs to actually be in data """
        logger.info("getLungs()")
        lungs = data3d < -300
        lungs[ fatlessbody == 0 ] = 0
        lungs = binaryFillHoles(lungs, z_axis=True)

        # remove all blobs that don't go through lower 1/4 of body
        logger.debug("remove all blobs that don't go through lower 1/4 of body")
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
        logger.debug("try to separate connected intestines")
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
        logger.debug("leave only lungs in data")
        lungs = skimage.measure.label(lungs, background=0)
        #ed = sed3.sed3(lungs); ed.show()
        unique, counts = np.unique(lungs, return_counts=True)
        unique = unique[1:]; counts = counts[1:] # remove background label (is 0)
        centroids = scipy.ndimage.measurements.center_of_mass(lungs, lungs, unique)
        if len(unique) == 0:
            logger.warning("Couldn't find lungs!")
            return np.zeros(data3d.shape, dtype=np.bool).astype(np.bool)

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
        logger.debug("remove trachea") # TODO - detect if there is one first
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
    def getDiaphragm(cls, data3d, spacing, lungs):
        """ Returns interpolated shape of Thoracic diaphragm (continues outsize of body) """
        logger.info("getDiaphragm()")
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
        diaphragm = np.zeros(diaphragm.shape, dtype=np.bool).astype(np.bool)
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
    def getKidneys(cls, data3d, spacing, fatlessbody, lungs_stats):
        """ between 150 and 300, cant go lower then 150 """
        logger.info("getKidneys()")
        spacing_vol = spacing[0]*spacing[1]*spacing[2]

        # work only on data under lungs -5cm
        lungs_end = lungs_stats["end"]
        lungs_end = int(max(0, lungs_end-(50/spacing[0])))
        orig_shape = tuple(data3d.shape)
        data3d = data3d[lungs_end:,:,:]; fatlessbody = fatlessbody[lungs_end:,:,:]

        #threshold tisue (without ribs)
        fatless_dst = scipy.ndimage.morphology.distance_transform_edt(fatlessbody, sampling=spacing) # for ignoring ribs
        kidneys = (fatless_dst > 15) & (data3d > cls.KIDNEYS_THRESHOLD_1); del(fatless_dst) # includes bones
        kidneys = binaryClosing(kidneys, structure=getSphericalMask([5,]*3, spacing=spacing))
        kidneys = binaryFillHoles(kidneys, z_axis=True)
        #ed = sed3.sed3(data3d, contour=kidneys); ed.show()

        ###

        # remove spine # TODO - redo
        bones_only  = (data3d > 500) # only rough bone contours
        b200_l = skimage.measure.label((data3d > cls.KIDNEYS_THRESHOLD_2), background=0) # TODO - check if threshold is ok
        tmp = b200_l.copy(); tmp[bones_only == 0] = 0
        bone_labels = np.unique(tmp)[1:]; del(tmp); del(bones_only)
        for l in np.unique(b200_l)[1:]:
            if l not in bone_labels:
                b200_l[b200_l == l] = 0
        kidneys[b200_l != 0] = 0

        # remove objects with too small volume
        kidneys = skimage.morphology.remove_small_objects(kidneys, min_size=int(cls.KIDNEYS_MIN_VOLUME/spacing_vol))

        # TODO - remove connected vessels

        # try to add kidney stones + middle part of kidneys #TODO
        kidneys = scipy.ndimage.binary_dilation(kidneys, structure=getSphericalMask([10,]*3, spacing=spacing))
        kidneys[(data3d > 150) == 0] = 0
        kidneys = skimage.morphology.remove_small_objects(kidneys, min_size=int(cls.KIDNEYS_MIN_VOLUME/spacing_vol))

        #ed = sed3.sed3(data3d, contour=kidneys); ed.show()
        # expand array to original shape
        kidneys_s = kidneys; kidneys = np.zeros(orig_shape, dtype=kidneys.dtype).astype(kidneys.dtype)
        kidneys[lungs_end:,:,:] = kidneys_s

        return kidneys

    @classmethod
    def getBones(cls, data3d, spacing, fatless, lungs, kidneys):
        """
        Good enough sgementation of all bones
        * data3d - everything, but body must be removed
        """
        logger.info("getBones()")
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

        # remove stuff connected to heart and kidneys
        if np.sum(lungs) != 0:
            wseeds = ( bones == 1 ).astype(np.int8) # = 1
            wseeds[ (fatless_dst < 15) & (fatless == 1) & (b200 == 1) ] = 1 # ribs readded
            wseeds[s][:,:int(lungs_hull.shape[1]/2),:][ lungs_hull[:,:int(lungs_hull.shape[1]/2),:] == 1 ] = 2
            wseeds[kidneys == 1] = 2
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
    def getAbdomen(cls, data3d, spacing, fatlessbody, diaphragm, bones_stats):
        """ Helpful for segmentation of organs in abdomen """
        logger.info("getAbdomen()")
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
    def getVessels(cls, data3d, spacing, bones, bones_stats, contrast_agent=True):
        """
        Tabular value of blood radiodensity is 13-50 HU.
        When contrast agent is used, it rises to roughly 100-140 HU.
        Vessels are segmentable only if contrast agent was used.
        """
        logger.info("getVessels()")
        points_spine = bones_stats["spine"]
        if len(points_spine) == 0:
            logger.warning("Couldn't find vessels!")
            return np.zeros(data3d.shape, dtype=np.bool).astype(np.bool)
        # get spine z-range
        spine_zmin = points_spine[0][0]; spine_zmax = points_spine[-1][0]

        #ed = sed3.sed3(data3d, contour=bones); ed.show()

        SPINE_WIDTH = int(cls.VESSELS_SPINE_WIDTH/spacing[2])
        SPINE_HEIGHT = int(cls.VESSELS_SPINE_HEIGHT/spacing[1])

        if contrast_agent:
            vessels = data3d > cls.VESSELS_THRESHOLD

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
                SPINE_HEIGHT = sc[0]-SPINE_HEIGHT

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
                    elif c[0] > SPINE_HEIGHT: continue # no lower then spine height
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
            return np.zeros(data3d.shape, dtype=np.bool).astype(np.bool)

            # TODO - try it anyway
            # - FIND EDGES, THRESHOLD EDGES - canny
            # - hough_circle
            # - convolution with kernel with very big z-axis
            # - combine last two steps
            # - points_spine - select close circles to spine

    @classmethod
    def getAorta(cls, data3d, spacing, vessels, vessels_stats):
        logger.info("getAorta()")
        points = vessels_stats["aorta"]
        if len(points) == 0 or np.sum(vessels) == 0:
            logger.warning("Couldn't find aorta volume!")
            return np.zeros(vessels.shape, dtype=np.bool).astype(np.bool)

        aorta = np.zeros(vessels.shape, dtype=np.bool).astype(np.bool)
        for p in points:
            aorta[p[0],p[1],p[2]] = 1
        aorta = growRegion(aorta, vessels, iterations=cls.VESSELS_AORTA_RADIUS)

        return aorta

    @classmethod
    def getVenaCava(cls, data3d, spacing, vessels, vessels_stats):
        logger.info("getVenaCava()")
        points = vessels_stats["vena_cava"]
        if len(points) == 0 or np.sum(vessels) == 0:
            logger.warning("Couldn't find venacava volume!")
            return np.zeros(vessels.shape, dtype=np.bool).astype(np.bool)

        venacava = np.zeros(vessels.shape, dtype=np.bool).astype(np.bool)
        for p in points:
            venacava[p[0],p[1],p[2]] = 1
        venacava = growRegion(venacava, vessels, iterations=cls.VESSELS_VENACAVA_RADIUS)

        return venacava

    ##################
    ### Statistics ###
    ##################

    @classmethod
    def analyzeLungs(cls, data3d, spacing, lungs):
        logger.info("analyzeLungs()")

        lungs_pad = getDataPadding(lungs)
        lungs_start = lungs_pad[0][0] # start of lungs on z-axis
        lungs_end = lungs.shape[0]-lungs_pad[0][1] # end of lungs on z-axis

        return {"start":lungs_start, "end":lungs_end}

    @classmethod
    def analyzeBones(cls, data3d, spacing, fatlessbody, bones, lungs_stats):
        """ Returns: {"spine":points_spine, "hip_joints":points_hip_joints, "hip_start":[]} """
        logger.info("analyzeBones()")

        # remove every bone higher then lungs
        lungs_start = lungs_stats["start"] # start of lungs on z-axis
        lungs_end = lungs_stats["end"] # end of lungs on z-axis
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
        points_hip_start_l = {}; points_hip_start_r = {}
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
            left_c = [None, None]; center_c = [None, None]; right_c = [None, None]
            if left_v > 0:
                left_c = list(scipy.ndimage.measurements.center_of_mass(left))
                left_c[1] = left_c[1]+pad[1][0]
            if center_v > 0:
                center_c = list(scipy.ndimage.measurements.center_of_mass(center))
                center_c[1] = center_c[1]+left_sep
            if right_v > 0:
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

            # try to detect hip bones start on z axis
            if (z >= lungs_end) and (left_v/total_v > 0.1):
                points_hip_start_l[z] = (z, int(left_c[0]), int(left_c[1]))
            if (z >= lungs_end) and (right_v/total_v > 0.1):
                points_hip_start_r[z] = (z, int(right_c[0]), int(right_c[1]))

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

        # try to detect start of hip bones
        points_hip_start = [None, None]
        end_z = bones.shape[0]-1 if len(points_hip_joints)==0 else points_hip_joints[0][0]
        for z in range(end_z, lungs_start, -1):
            if z not in points_hip_start_l:
                if (z+1) in points_hip_start_l:
                    points_hip_start[0] = points_hip_start_l[z+1]
                break
        for z in range(end_z, lungs_start, -1):
            if z not in points_hip_start_r:
                if (z+1) in points_hip_start_r:
                    points_hip_start[1] = points_hip_start_r[z+1]
                break

        # seeds = np.zeros(bones.shape)
        # for p in points_spine_c: seeds[p[0], p[1], p[2]] = 2
        # for p in points_spine: seeds[p[0], p[1], p[2]] = 1
        # for p in points_hip_joints_l: seeds[p[0], p[1], p[2]] = 2
        # for p in points_hip_joints_r: seeds[p[0], p[1], p[2]] = 2
        # for p in points_hip_joints: seeds[p[0], p[1], p[2]] = 3
        # seeds = scipy.ndimage.morphology.grey_dilation(seeds, size=(1,5,5))
        # ed = sed3.sed3(data3d, contour=bones, seeds=seeds); ed.show()

        return {"spine":points_spine, "hip_joints":points_hip_joints, "hip_start":points_hip_start}

    @classmethod
    def analyzeVessels(cls, data3d, spacing, vessels, bones_stats):
        """ Returns: {"aorta":[], "vena_cava":[]} """
        logger.info("analyzeVessels()")
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
