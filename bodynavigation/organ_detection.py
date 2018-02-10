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
import io

import numpy as np
import scipy
import scipy.ndimage
import skimage.measure
import skimage.transform
import skimage.morphology
import skimage.segmentation
import skimage.feature

import sed3 # for testing

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
    out = np.zeros(full_shape) + padding_value
    s = []
    for dim in range(len(data.shape)):
        s.append( slice( pads[dim][0], out.shape[dim]-pads[dim][1] ) )
    out[tuple(s)] = data
    return out

#
# Main class
#

class OrganDetection(object):
    """
    * getORGAN()
        - for when you create class object (standard usage)
        - resizes output to corect shape (unless you use "resized=False")
        - saves output for future calls
    * _getORGAN()
        - for using algorithm directly without class object: OrganDetection._getORGAN()
        - does not resize output
        - does not save output for future calls
        - all required data is in function parameters
    * internal voxelsize is self.spacing and is first normalized and then scaled to [normed-z, 1, 1]
    """

    NORMED_FATLESS_BODY_SIZE = [200,300] # normalized size of fatless body on [Y,X] in mm
    # [189.8, 251.7] - 3Dircadb1.1
    # [192.4, 268.2] - 3Dircadb1.2
    # [205.1, 307.6] - imaging.nci.nih.gov_NSCLC-Radiogenomics-Demo_1.3.6.1.4.1.14519.5.2.1.4334.1501.332134560740628899985464129848
    # [220.3, 325.2] - imaging.nci.nih.gov_NSCLC-Radiogenomics-Demo_1.3.6.1.4.1.14519.5.2.1.4334.1501.117528645891554472837507616577

    def __init__(self, data3d=None, voxelsize=[1,1,1], size_normalization=True, rescale=True):
        """
        * Values of input data should be in HU units (or relatively close). [air -1000, water 0]
            https://en.wikipedia.org/wiki/Hounsfield_scale
        * All coordinates and sizes are in [Z,Y,X] format
        * Expecting data3d to be corectly oriented
        * Voxel size is in mm
        """

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

        # compressed masks - this lowered memory usage to 0.042% for bones
        self.masks_comp = {
            "body":None,
            "fatlessbody":None,
            "bones":None,
            "lungs":None,
            }

    @classmethod
    def fromReadyData(cls, data3d, spacing, body=None, fatlessbody=None, bones=None, lungs=None ):
        """ For super fast testing """
        obj = cls()

        obj.data3d = data3d
        obj.spacing = spacing
        obj.cut_shape = data3d.shape
        obj.padding = [[0,0],[0,0],[0,0]]

        obj.orig_voxelsize = np.asarray(spacing, dtype=np.float)
        obj.orig_coord_scale = np.asarray([1,1,1], dtype=np.float)
        obj.orig_coord_intercept = np.asarray([0,0,0], dtype=np.float)

        if body is not None: obj.masks_comp["body"] = compressArray(body)
        if fatlessbody is not None: obj.masks_comp["fatlessbody"] = compressArray(fatlessbody)
        if bones is not None: obj.masks_comp["bones"] = compressArray(bones)
        if lungs is not None: obj.masks_comp["lungs"] = compressArray(lungs)

        return obj

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

        # cut off empty parts of data
        body = self._getBody(data3d, voxelsize)
        data3d[ body == 0 ] = -1024
        padding = getDataPadding(body)
        data3d = cropArray(data3d, padding)
        body = cropArray(body, padding)
        cut_shape = data3d.shape # without padding

        # size norming based on body size on xy axis
        # this just recalculates voxelsize (doesnt touch actual data)
        if size_normalization:
            fatlessbody = self._getFatlessBody(data3d, voxelsize, body)
            size_v = [ fatlessbody.shape[dim+1]-np.sum(pad) for dim, pad in enumerate(getDataPadding(fatlessbody[int(fatlessbody.shape[0]/2),:,:])) ]
            size_mm = [ size_v[0]*voxelsize[1], size_v[1]*voxelsize[2] ] # fatlessbody size in mm on X and Y axis
            size_scale = [ None, self.NORMED_FATLESS_BODY_SIZE[0]/size_mm[0], self.NORMED_FATLESS_BODY_SIZE[1]/size_mm[1] ]
            size_scale[0] = (size_scale[1]+size_scale[2])/2 # scaling on z-axis is average of scaling on x,y-axis
            voxelsize = [
                voxelsize[0]*size_scale[0],
                voxelsize[1]*size_scale[1],
                voxelsize[2]*size_scale[2],
                ]
        del(body) # not needed anymore

        # resize data on x,y axis (upscaling creates ghosting effect on z-axis)
        if not rescale:
            spacing = voxelsize
        else:
            new_shape = np.asarray([ data3d.shape[0], data3d.shape[1] * voxelsize[1], \
                data3d.shape[2] * voxelsize[2] ]).astype(np.int)
            spacing = np.asarray([ voxelsize[0], 1, 1 ])
            data3d = skimage.transform.resize(
                data3d, new_shape, order=3, mode="reflect", clip=True, preserve_range=True,
                ).astype(np.int16)

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
        out = padArray(out, self.padding, padding_value=padding_value)
        return out

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
            elif part == "bones":
                data = self._getBones(self.data3d, self.spacing, self.getFatlessBody(raw=True))
            elif part == "lungs":
                data = self._getLungs(self.data3d, self.spacing, self.getBody(raw=True))

            self.masks_comp[part] = compressArray(data)

        if not raw: data = self.toOutputFormat(data)
        return data

    def getBody(self, raw=False):
        return self.getPart("body", raw=raw)

    def getFatlessBody(self, raw=False):
        return self.getPart("fatlessbody", raw=raw)

    def getBones(self, raw=False):
        return self.getPart("bones", raw=raw)

    def getLungs(self, raw=False):
        return self.getPart("lungs", raw=raw)

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
    def _getFatlessBody(cls, data3d, spacing, body):
        """
        Returns convex hull of body without fat and skin
        """
        logger.info("_getFatlessBody")
        # remove fat
        fatless = (data3d > 0)
        fatless[ (body == 1) & (data3d < -500) ] = 1 # body cavities
        fatless = scipy.ndimage.morphology.binary_opening(fatless, structure=getSphericalMask([5,5,5], spacing=spacing))
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
    def _getBones(cls, data3d, spacing, fatless, graphcut=False):
        """
        Good enough sgementation of all bones
        * data3d - everything, but body must be removed
        * graphcut - very good on some data, but can eat RAM by 10s of GB.
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

        #ed = sed3.sed3(data3d, contour=bones); ed.show()

        # segmentation > 200, save only objects that are connected from > 300
        b200 = skimage.measure.label((data3d > 200), background=0)
        seeds_l = b200.copy(); seeds_l[ bones == 0 ] = 0
        for l in np.unique(seeds_l)[1:]:
            bones[ b200 == l ] = 1
        del(b200); del(seeds_l)
        bones = binaryClosing(bones, structure=getSphericalMask([5,]*3, spacing=spacing))
        bones = binaryFillHoles(bones, z_axis=True)

        #ed = sed3.sed3(data3d, contour=(b200 == 1), seeds=bones); ed.show()

        # TODO - zkusit graphcut po prekrivajicich se castech (mensi spotreba RAM) - jako vysledek vybrat MODE segmentace kazdeho pixelu
        if graphcut:
            import pysegbase.pycut as pspc # 3D graphcut with seeds

            # create labeled seeds for graphcut
            label = np.zeros(data3d.shape, dtype=np.int8)
            label[ scipy.ndimage.morphology.distance_transform_edt(bones == 0, sampling=spacing) > 15 ] = 2

            # no seeds where could be ribs
            # fatless = skimage.morphology.dilation(fatless, getSphericalMask([3,3,3])) # some padding for safety
            # label[ (fatless_dst < 15) & (fatless == 1) ] = 0
            label[ fatless == 0 ] = 2

            label[ bones == 1 ] = 1

            #ed = sed3.sed3(data3d, seeds=label); ed.show()

            # graphcut - works great
            igc = pspc.ImageGraphCut(data3d, voxelsize=spacing)
            igc.set_seeds(label)
            igc.run()
            #igc.interactivity() # works only if sed3 was not used
            bones = igc.segmentation == 0

            # ed = sed3.sed3(data3d, contour=bones); ed.show()

            bones = binaryClosing(bones, structure=getSphericalMask([5,]*3, spacing=spacing)) # TODO - prekontrolovat vysledek - predtim nefugovalo spravne

            # ed = sed3.sed3(data3d, seeds=(label==1), contour=bones); ed.show()
            # ed = sed3.sed3(data3d, seeds=bones); ed.show()

        return bones

    @classmethod
    def _getLungs(cls, data3d, spacing, body):
        """ Expects lungs to actually be in data """
        logger.info("_getLungs")
        lungs = data3d < -500
        lungs[ body == 0 ] = 0
        lungs = binaryFillHoles(lungs, z_axis=True)

        # leave only biggest object in data (2 if they are about the same size)
        lungs = skimage.measure.label(lungs, background=0)
        unique, counts = np.unique(lungs, return_counts=True)
        unique = unique[1:]; counts = counts[1:] # remove background label (is 0)
        # get 2 biggest blobs
        idx_1st = list(counts).index(max(counts))
        count_1st = counts[idx_1st]; counts[idx_1st] = 0
        idx_2nd = list(counts).index(max(counts))
        count_2nd = counts[idx_2nd]; counts[idx_1st] = count_1st
        # leave only lungs in data
        lungs[ lungs == unique[idx_1st] ] = -1
        if abs(count_1st-count_2nd)/(count_1st+count_2nd) < 0.3: # if volume diff is lower then 30%
            lungs[ lungs == unique[idx_2nd] ] = -1
        lungs = lungs == -1

        return lungs

    ################

    def analyzeBones(self, raw=False):
        """ Returns: points_spine, points_hip_joint """
        logger.info("analyzeBones")

        fatlessbody = self.getFatlessBody(raw=True)
        bones = self.getBones(raw=True)
        lungs = self.getLungs(raw=True)

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
            structure=getSphericalMask([20,]*3, spacing=self.spacing)) # takes around 1m

        #ed = sed3.sed3(self.data3d, contour=bones); ed.show()

        points_spine = []
        points_hip_joint_l = []; points_hip_joint_r = []
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
            if (left_v/total_v < 0.2) or (right_v/total_v < 0.2):
                points_spine.append( (z, int(center_c[0]), int(center_c[1])) )

            # try to detect hip joints
            if (z >= lungs_end) and (left_v/total_v > 0.4) and (right_v/total_v > 0.4):
                # gets also leg bones
                #print(z, abs(left_c[1]-right_c[1]))
                if abs(left_c[1]-right_c[1]) < 180:
                    # anything futher out should be only leg bones
                    points_hip_joint_l.append( (z, int(left_c[0]), int(left_c[1])) )
                    points_hip_joint_r.append( (z, int(right_c[0]), int(right_c[1])) )

        # calculate centroid of hip points
        points_hip_joint = []
        if len(points_hip_joint_l) != 0:
            z, y, x = zip(*points_hip_joint_l); l = len(z)
            cl = (int(sum(z)/l), int(sum(y)/l), int(sum(x)/l))
            z, y, x = zip(*points_hip_joint_r); l = len(z)
            cr = (int(sum(z)/l), int(sum(y)/l), int(sum(x)/l))
            points_hip_joint = [cl, cr]

        # remove any spine points under detected hips
        if len(points_hip_joint) != 0:
            newp = []
            for p in points_spine:
                if p[0] < points_hip_joint[0][0]:
                    newp.append(p)
            points_spine = newp

        # fit curve to spine points and recalculate new points from curve
        z, y, x = zip(*points_spine)
        z_new = list(range(z[0], z[-1]+1))

        zz1 = np.polyfit(z, y, 3)
        f1 = np.poly1d(zz1)
        y_new = f1(z_new)

        zz2 = np.polyfit(z, x, 3)
        f2 = np.poly1d(zz2)
        x_new = f2(z_new)

        points_spine = [ tuple([int(z_new[i]), int(y_new[i]), int(x_new[i])]) for i in range(len(z_new)) ]

        # seeds = np.zeros(bones.shape)
        # for p in points_spine: seeds[p[0], p[1], p[2]] = 1
        # for p in points_hip_joint_l: seeds[p[0], p[1], p[2]] = 2
        # for p in points_hip_joint_r: seeds[p[0], p[1], p[2]] = 2
        # for p in points_hip_joint: seeds[p[0], p[1], p[2]] = 3
        # seeds = scipy.ndimage.morphology.grey_dilation(seeds, size=(1,5,5))
        # ed = sed3.sed3(self.data3d, contour=bones, seeds=seeds); ed.show()

        if not raw:
            points_spine = [ tuple(self.toOutputCoordinates(p).astype(np.int)) for p in points_spine ]
            points_hip_joint = [ tuple(self.toOutputCoordinates(p).astype(np.int)) for p in points_hip_joint ]

        return points_spine, points_hip_joint


if __name__ == "__main__":
    import argparse
    import io3d, sed3

    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # input parser
    parser = argparse.ArgumentParser(description="Organ Detection")
    parser.add_argument('-i','--datadir', default=None,
            help='path to data dir')
    parser.add_argument('-r','--readydir', default=None,
            help='path to ready data dir (for testing)')
    parser.add_argument("--dump", action="store_true",
            help='dump all data to ready_data dir and exit')
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
        print("Loading all data from ready_dir")
        data3d_p = os.path.join(args.readydir, "data3d.dcm")
        body_p = os.path.join(args.readydir, "body.dcm")
        fatlessbody_p = os.path.join(args.readydir, "fatlessbody.dcm")
        bones_p = os.path.join(args.readydir, "bones.dcm")
        lungs_p = os.path.join(args.readydir, "lungs.dcm")

        data3d, metadata = io3d.datareader.read(data3d_p)
        spacing = metadata["voxelsize_mm"]

        body = None
        fatlessbody = None
        bones = None
        lungs = None

        if os.path.exists(body_p):
            body, _ = io3d.datareader.read(body_p)
            body = body.astype(np.bool)
        if os.path.exists(fatlessbody_p):
            fatlessbody, _ = io3d.datareader.read(fatlessbody_p)
            fatlessbody = fatlessbody.astype(np.bool)
        if os.path.exists(bones_p):
            bones, _ = io3d.datareader.read(bones_p)
            bones = bones.astype(np.bool)
        if os.path.exists(lungs_p):
            lungs, _ = io3d.datareader.read(lungs_p)
            lungs = lungs.astype(np.bool)

        obj = OrganDetection.fromReadyData(data3d, spacing, \
            body=body, fatlessbody=fatlessbody, bones=bones, lungs=lungs )

        del(body)
        del(fatlessbody)
        del(bones)
        del(lungs)

    if args.dump:
        print("Dumping all data to ready_dir")
        readydir = "./ready_dir"
        if not os.path.exists(readydir): os.makedirs(readydir)

        data3d_p = os.path.join(readydir, "data3d.dcm")
        body_p = os.path.join(readydir, "body.dcm")
        fatlessbody_p = os.path.join(readydir, "fatlessbody.dcm")
        bones_p = os.path.join(readydir, "bones.dcm")
        lungs_p = os.path.join(readydir, "lungs.dcm")

        data3d = obj.data3d
        spacing = list(obj.spacing)
        io3d.datawriter.write(data3d, data3d_p, 'dcm', {'voxelsize_mm': spacing})

        # note: Masks look wierd when opened in ImageJ

        body = obj.getBody(raw=True).astype(np.int8)
        io3d.datawriter.write(body, body_p, 'dcm', {'voxelsize_mm': spacing})
        del(body)

        fatlessbody = obj.getFatlessBody(raw=True).astype(np.int8)
        io3d.datawriter.write(fatlessbody, fatlessbody_p, 'dcm', {'voxelsize_mm': spacing})
        del(fatlessbody)

        bones = obj.getBones(raw=True).astype(np.int8)
        io3d.datawriter.write(bones, bones_p, 'dcm', {'voxelsize_mm': spacing})
        del(bones)

        lungs = obj.getLungs(raw=True).astype(np.int8)
        io3d.datawriter.write(lungs, lungs_p, 'dcm', {'voxelsize_mm': spacing})
        del(lungs)

        sys.exit(0)

    #########
    print("-----------------------------------------------------------")

    body = obj.getBody()
    fatlessbody = obj.getFatlessBody()
    bones = obj.getBones()
    # lungs = obj.getLungs()

    # ed = sed3.sed3(data3d, contour=body); ed.show()
    # ed = sed3.sed3(data3d, contour=fatlessbody); ed.show()
    # ed = sed3.sed3(data3d, contour=bones); ed.show()
    # ed = sed3.sed3(data3d, contour=lungs); ed.show()

    points_spine, points_hip_joint = obj.analyzeBones()

    seeds = np.zeros(bones.shape)
    for p in points_spine: seeds[p[0], p[1], p[2]] = 1
    for p in points_hip_joint: seeds[p[0], p[1], p[2]] = 2
    seeds = scipy.ndimage.morphology.grey_dilation(seeds, size=(1,5,5))
    ed = sed3.sed3(data3d, contour=bones, seeds=seeds); ed.show()








