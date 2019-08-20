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

# run with: "python -m bodynavigation.organ_detection -h"
from .tools import (
    getSphericalMask,
    getDiskMask,
    binaryClosing,
    binaryFillHoles,
    getDataPadding,
    cropArray,
    padArray,
    polyfit3D,
    regionGrowing,
    getDataFractions,
    getBiggestObjects,
    firstNonzero,
)


class OrganDetectionAlgo(object):
    """
    Container for segmentation and analysis algorithms used by OrganDetection class.

    For constants in class: (placed just before function defs)
        tresholds are in HU
        sizes are in mm
        areas are in mm2
        volumes are in mm3
    """

    @classmethod
    def cleanData(cls, data3d, spacing, remove_brightness_errors=True):
        """
        Filters out noise, removes some errors in data, sets undefined voxel value to -1024, etc ...
        """
        logger.info("cleanData()")
        # fix for io3d <-512;511> value range bug, that is caused by hardcoded slope 0.5 in dcmreader
        if np.min(data3d) >= -512:
            logger.debug("Fixing io3d <-512;511> value range bug")
            data3d = data3d * 2

        # set padding value to -1024 (undefined voxel values in space outside of senzor range)
        logger.debug("Setting 'padding' value")
        data3d[data3d == data3d[0, 0, 0]] = -1024

        # limit value range to <-1024;int16_max> so it can fit into int16
        # [ data3d < -1024 ] => less dense then air - padding values
        # [ data3d > int16_max  ] => int16_max
        logger.debug("Converting to int16")
        data3d[data3d < -1024] = -1024
        data3d[data3d > np.iinfo(np.int16).max] = np.iinfo(np.int16).max
        data3d = data3d.astype(np.int16)

        # filter out noise - median filter with radius 1 (kernel 1x3x3)
        # Filter is not used along z axis because the slices are so thick that any filter will
        # create ghosts of pervous and next slices on them.
        logger.debug("Removing noise with filter")
        for z in range(data3d.shape[0]):
            data3d[z, :, :] = scipy.ndimage.filters.median_filter(data3d[z, :, :], 3)
        # ed = sed3.sed3(data3d); ed.show()

        # remove high brightness errors near edges of valid data (takes about 70s)
        if remove_brightness_errors:
            logger.debug(
                "Removing high brightness errors near edges of valid data"
            )  # TODO - clean this part up
            valid_mask = data3d > -1024
            valid_mask = skimage.measure.label(valid_mask, background=0)
            unique, counts = np.unique(valid_mask, return_counts=True)
            unique = unique[1:]
            counts = counts[1:]  # remove background label (is 0)
            valid_mask = valid_mask == unique[list(counts).index(max(counts))]
            for z in range(valid_mask.shape[0]):
                tmp = valid_mask[z, :, :]
                if np.sum(tmp) == 0:
                    continue
                tmp = skimage.morphology.convex_hull_image(tmp)
                # get contours
                tmp = skimage.feature.canny(tmp) != 0
                # thicken contour (expecting 512x512 resolution)
                tmp = scipy.ndimage.binary_dilation(
                    tmp, structure=skimage.morphology.disk(11, dtype=np.bool)
                )
                # lower all values near border bigger then BODY_THRESHOLD closer to BODY_THRESHOLD
                dst = scipy.ndimage.morphology.distance_transform_edt(tmp).astype(
                    np.float
                )
                dst = dst / np.max(dst)
                dst[dst != 0] = 0.01 ** dst[dst != 0]
                dst[dst == 0] = 1.0

                mask = data3d[z, :, :] > cls.BODY_THRESHOLD
                data3d[z, :, :][mask] = (
                    ((data3d[z, :, :][mask].astype(np.float) + 300) * dst[mask]) - 300
                ).astype(
                    np.int16
                )  # TODO - use cls.BODY_THRESHOLD
            del (valid_mask, dst)
            # ed = sed3.sed3(data3d); ed.show()

        # remove anything that is not in body volume
        logger.debug("Removing all data outside of segmented body")
        body = cls.getBody(data3d, spacing)
        data3d[body == 0] = -1024

        # ed = sed3.sed3(data3d); ed.show()
        return data3d, body

    @classmethod
    def dataRegistrationPoints(
        cls, spacing, body, fatlessbody, lungs_stats, bones_stats
    ):
        """
        How to use:
            1. Create OrganDetection object with notransformation mode
            2. use this function to get reg_points
            3. Create new OrganDetection with transformation and reg_points as input
            4. transform masks and data from old OrganDetection to new one
            -. Only reuse body,fatless,lungs masks; Rest might need to be recalculated on registred data3d
        """
        logger.info("dataRegistrationPoints()")
        reg_points = {}

        # body shape, spacing and padding
        reg_points["shape"] = body.shape
        reg_points["spacing"] = spacing  # so we can recalculate to mm later
        reg_points["padding"] = getDataPadding(body)

        # scaling on z axis
        reg_points["lungs_end"] = lungs_stats["end"]
        if len(bones_stats["hips_start"]) == 0:
            logger.warning(
                "Since no 'hips_start' points were found, using shape[0] as registration point."
            )
            reg_points["hips_start"] = fatlessbody.shape[0]
        else:
            reg_points["hips_start"] = int(
                np.average([p[0] for p in bones_stats["hips_start"]])
            )

        # precalculate sizes and centroids at lungs_end:hips_start
        widths = []
        heights = []
        centroids = []
        for z in range(reg_points["lungs_end"], reg_points["hips_start"]):
            if np.sum(fatlessbody[z, :, :]) == 0:
                continue
            spads = getDataPadding(fatlessbody[z, :, :])
            heights.append(fatlessbody[z, :, :].shape[0] - np.sum(spads[0]))
            widths.append(fatlessbody[z, :, :].shape[1] - np.sum(spads[1]))
            centroids.append(
                scipy.ndimage.measurements.center_of_mass(fatlessbody[z, :, :])
            )

        # scaling on x,y axes # TODO - maybe save multiple values (at least 3) between lungs_end:hips_start -> warp transform
        if len(widths) == 0:
            raise Exception(
                "Could not calculate abdomen sizes! No fatlessbody between lungs_end:hips_start!"
            )
        else:
            reg_points["fatlessbody_height"] = np.median(heights)
            reg_points["fatlessbody_width"] = np.median(widths)

        # relative centroid (to array shape)
        centroids_arr = np.zeros((len(centroids), 2), dtype=np.float)
        for i in range(len(centroids)):
            centroids_arr[i, :] = np.asarray(centroids[i], dtype=np.float)
        centroid = np.median(centroids_arr, axis=0)
        reg_points["fatlessbody_centroid"] = tuple(
            centroid / np.array(fatlessbody[z, :, :].shape, dtype=np.float)
        )

        logger.debug(reg_points)
        return reg_points

    ####################
    ### Segmentation ###
    ####################

    BODY_THRESHOLD = -300

    @classmethod
    def getBody(cls, data3d, spacing):
        """
        Input: noiseless data3d
        Returns binary mask representing body volume (including most cavities)

        Needs to work on raw cleaned data!
        """
        logger.info("getBody()")
        # segmentation of body volume
        body = (data3d > cls.BODY_THRESHOLD).astype(np.bool)

        # fill holes
        body = binaryFillHoles(body, z_axis=True)

        # binary opening
        body = scipy.ndimage.morphology.binary_opening(
            body, structure=getSphericalMask(5, spacing=spacing)
        )

        # leave only biggest object in data
        body = getBiggestObjects(body, N=1)

        # filling nose/mouth openings + connected cavities
        # - fills holes separately on every slice along z axis (only part of mouth and nose should have cavity left)
        for z in range(body.shape[0]):
            body[z, :, :] = binaryFillHoles(body[z, :, :])

        return body

    FATLESSBODY_THRESHOLD = 20
    FATLESSBODY_AIR_THRESHOLD = -300

    @classmethod
    def getFatlessBody(
        cls, data3d, spacing, body
    ):  # TODO - ignore nipples (and maybe belly button) when creating convex hull
        """
        Returns convex hull of body without fat and skin

        Needs to work on raw cleaned data!
        """
        logger.info("getFatlessBody()")
        # remove fat
        fatless = data3d > cls.FATLESSBODY_THRESHOLD
        fatless = scipy.ndimage.morphology.binary_opening(
            fatless, structure=getSphericalMask(5, spacing=spacing)
        )  # remove small segmentation errors
        # fill body cavities, but ignore air near borders of body
        body_border = body & (
            scipy.ndimage.morphology.binary_erosion(
                body, structure=getDiskMask(10, spacing=spacing)
            )
            == 0
        )
        fatless[
            (data3d < cls.FATLESSBODY_AIR_THRESHOLD) & (body_border == 0) & (body == 1)
        ] = 1
        # remove skin
        tmp = scipy.ndimage.morphology.binary_opening(
            fatless, structure=getSphericalMask(7, spacing=spacing)
        )
        fatless[body_border] = tmp[body_border]
        # ed = sed3.sed3(data3d, contour=fatless, seeds=body_border); ed.show()
        # save convex hull along z-axis
        for z in range(fatless.shape[0]):
            bsl = skimage.measure.label(body[z, :, :], background=0)
            for l in np.unique(bsl)[1:]:
                tmp = fatless[z, :, :] & (bsl == l)
                if np.sum(tmp) == 0:
                    continue
                fatless[z, :, :][skimage.morphology.convex_hull_image(tmp) == 1] = 1
                fatless[z, :, :][body[z, :, :] == 0] = 0
        return fatless

    LUNGS_THRESHOLD = -300
    LUNGS_INTESTINE_SEGMENTATION_OFFSET = 20  # in mm
    LUNGS_TRACHEA_MAXWIDTH = 40  # from side to side

    @classmethod
    def getLungs(cls, data3d, spacing, fatlessbody):
        """
        Expects lungs to actually be in data

        Needs to work on raw cleaned data!
        """
        logger.info("getLungs()")
        lungs = data3d < cls.LUNGS_THRESHOLD
        lungs[fatlessbody == 0] = 0
        lungs = binaryFillHoles(lungs, z_axis=True)

        # centroid of lungs, useful later.
        # (Anything up cant be intestines, calculated only from largest blob)
        logger.debug("get rough lungs centroid")
        lungs = skimage.measure.label(lungs, background=0)
        if (
            np.sum(lungs[0, :, :]) != 0
        ):  # if first slice has lungs (high chance of abdomen only data)
            # 'connect' lungs blobs that are on first slice
            # (this should fix any problems with only small sections of lungs in data)
            unique = np.unique(lungs)[1:]
            for u in unique:
                lungs[lungs == u] = unique[0]
        unique, counts = np.unique(lungs, return_counts=True)
        unique = unique[1:]
        counts = counts[1:]
        largest_id = unique[list(counts).index(max(counts))]
        centroid_z = int(
            scipy.ndimage.measurements.center_of_mass(lungs == largest_id)[0]
        )
        lungs = lungs != 0

        # try to separate connected intestines
        logger.debug("try to separate connected intestines")
        seeds = np.zeros(data3d.shape, dtype=np.int8)
        intestine_offset = int(cls.LUNGS_INTESTINE_SEGMENTATION_OFFSET / spacing[0])
        for z in range(data3d.shape[0]):
            if np.sum(lungs[z, :, :]) == 0:
                continue

            frac = [{"h": (2 / 3, 1), "w": (0, 1)}, {"h": (0, 2 / 3), "w": (0, 1)}]
            lower1_3, upper2_3 = getDataFractions(
                lungs[z, :, :], fraction_defs=frac, mask=fatlessbody[z, :, :]
            )  # views of lungs array
            lower1_3_s, upper2_3_s = getDataFractions(
                seeds[z, :, :], fraction_defs=frac, mask=fatlessbody[z, :, :]
            )  # views of seed array
            lower1_3_sum = np.sum(lower1_3)
            upper2_3_sum = np.sum(upper2_3)

            if (lower1_3_sum != 0) and (np.sum(seeds == 2) == 0):
                # lungs
                # IF: in lower 1/3 of body AND not after intestines
                lower1_3_s[lower1_3 != 0] = 1

            elif (
                (z > centroid_z)
                and (
                    np.sum(seeds[max(0, z - intestine_offset) : (z + 1), :, :] == 1)
                    == 0
                )
                and (lower1_3_sum == 0)
                and (upper2_3_sum != 0)
            ):
                # intestines or other non-lungs cavities
                # IF: slice is under centroid of lungs, has minimal offset from any detected lungs,
                #     stuff only in upper 2/3 of body
                upper2_3_s[upper2_3 != 0] = 2
        # ed = sed3.sed3(data3d, contour=lungs, seeds=seeds); ed.show()
        # using watershed region growing mode, because the thin tissue wall separating lungs and
        # intestines is enough to stop algorithm going to wrong side. "random_walker" would
        # work more reliable, but is more momory heavy.
        seeds = regionGrowing(data3d, seeds, lungs, mode="watershed")
        # ed = sed3.sed3(data3d, contour=lungs, seeds=seeds); ed.show()
        lungs = seeds == 1
        del seeds

        if np.sum(lungs) == 0:
            logger.warning("Couldn't find lungs!")
            return lungs

        # remove trachea (only the part sticking out)
        logger.debug("remove trachea")
        pads = getDataPadding(lungs)
        lungs_depth_mm = (lungs.shape[0] - pads[0][1] - pads[0][0]) * spacing[0]
        # try to remove only if lungs are longer then 200 mm on z-axis (not abdomen-only data)
        if lungs_depth_mm > 200:
            trachea_start_z = None
            max_width = 0
            for z in range(lungs.shape[0] - 1, 0, -1):
                if np.sum(lungs[z, :, :]) == 0:
                    continue
                pad = getDataPadding(lungs[z, :, :])
                width = lungs[z, :, :].shape[1] - (pad[1][0] + pad[1][1])

                if max_width <= width:
                    max_width = width
                    trachea_start_z = None
                elif (trachea_start_z is None) and (
                    width * spacing[1] < cls.LUNGS_TRACHEA_MAXWIDTH
                ):
                    trachea_start_z = z

            if trachea_start_z is not None:
                lungs[: min(lungs.shape[0], trachea_start_z + 1), :, :] = 0

        # return only blobs that have volume at centroid slice
        lungs = skimage.measure.label(lungs, background=0)
        unique = np.unique(lungs[centroid_z, :, :])[1:]
        for u in unique:
            lungs[lungs == u] = -1
        lungs = lungs == -1

        return lungs

    BONES_THRESHOLD_LOW = 200
    BONES_THRESHOLD_HIGH = 300
    BONES_RIBS_MAX_DEPTH = 15  # mm; max depth of ribs from surface of fatless body
    BONES_SHALLOW_BONES_MAX_DEPTH = (
        20
    )  # mm; bones that are in shallow depth (of fatless body), used for detection of end of ribs and start of hips
    BONES_LOW_MAX_DST = (
        15
    )  # mm; max distance of low thresholded bones from high thresholded parts

    @classmethod
    def getBones(
        cls, data3d, spacing, fatlessbody, lungs, lungs_stats
    ):  # TODO - pull more constants outside of function
        """
        Needs to work on raw cleaned data!

        Algorithm aims at not segmenting wrong parts over complete segmentation.
        """
        logger.info("getBones()")
        spacing_vol = spacing[0] * spacing[1] * spacing[2]
        fatlessbody_dst = scipy.ndimage.morphology.distance_transform_edt(
            fatlessbody, sampling=spacing
        )

        # create convex hull of lungs
        lungs_hull = np.zeros(lungs.shape, dtype=np.bool).astype(np.bool)
        for z in range(lungs_stats["start"], lungs_stats["end"]):
            if np.sum(lungs[z, :, :]) == 0:
                continue
            lungs_hull[z, :, :] = skimage.morphology.convex_hull_image(lungs[z, :, :])

        ### Basic high segmentation
        logger.debug("Basic high threshold segmentation")
        bones = data3d > cls.BONES_THRESHOLD_HIGH
        bones = binaryFillHoles(bones, z_axis=True)
        bones = skimage.morphology.remove_small_objects(
            bones.astype(np.bool), min_size=int((10 ** 3) / spacing_vol)
        )
        # readd segmented points that are in expected ribs volume
        bones[
            (fatlessbody_dst < cls.BONES_RIBS_MAX_DEPTH)
            & (data3d > cls.BONES_THRESHOLD_HIGH)
        ] = 1

        ### Remove errors of basic segmentation / create seeds
        logger.debug("Remove errors of basic segmentation / create seeds")
        bones = bones.astype(np.int8)  # use for seeds

        # remove possible segmented heart parts (remove upper half of convex hull of lungs)
        # ed = sed3.sed3(data3d, contour=lungs); ed.show()
        if np.sum(lungs_hull) != 0:
            # sometimes parts of ribs are slightly inside of lungs hull -> (making hull a bit smaller)
            lungs_hull_eroded = scipy.ndimage.binary_erosion(
                lungs_hull, structure=getDiskMask(10, spacing=spacing)
            )
            # get lungs height
            pads = getDataPadding(lungs_hull_eroded)
            lungs_hull_height = data3d.shape[1] - pads[1][0] - pads[1][1]
            # remove anything in top half of lungs hull
            remove_height = pads[1][0] + int(lungs_hull_height * 0.5)
            view_lungs_hull_top = lungs_hull_eroded[:, :remove_height, :]
            view_bones_top = bones[:, :remove_height, :]
            view_bones_top[(view_bones_top == 1) & view_lungs_hull_top] = 2

        # define sizes of spine and hip sections
        frac_left = {"h": (0, 1), "w": (0, 0.40)}
        frac_spine = {"h": (0.25, 1), "w": (0.40, 0.60)}
        frac_front = {"h": (0, 0.25), "w": (0.40, 0.60)}
        frac_right = {"h": (0, 1), "w": (0.60, 1)}

        # get ribs and and hips start index
        b_surface = (bones == 1) & (fatlessbody_dst < cls.BONES_SHALLOW_BONES_MAX_DEPTH)
        for z in range(
            data3d.shape[0]
        ):  # only left and right sections for ribs and hips detection
            view_spine, view_front = getDataFractions(
                b_surface[z, :, :],
                fraction_defs=[frac_spine, frac_front],
                mask=fatlessbody[z, :, :],
            )
            view_spine[:, :] = 0
            view_front[:, :] = 0  # TODO - test if changes original array
        b_surface_sums = np.sum(np.sum(b_surface, axis=1), axis=1)

        if np.sum(b_surface_sums[lungs_stats["end"] :] == 0) == 0:
            logger.warning("End of ribs not found in data! Using data3d.shape[0]")
            ribs_end = data3d.shape[0]
        else:
            ribs_end = lungs_stats["end"] + np.argmax(
                b_surface_sums[lungs_stats["end"] :] == 0
            )

        if (ribs_end == data3d.shape[0]) or (
            np.sum(b_surface_sums[min(data3d.shape[0], ribs_end + 1) :]) == 0
        ):
            logger.warning("Start of hips not found in data! Using data3d.shape[0]")
            hips_start = data3d.shape[0]
        else:
            rough_hips_start = (ribs_end + 1) + np.argmax(
                b_surface_sums[ribs_end + 1 :]
            )
            # go backwards by slices, until there is no voxels with high threshold in left or right sections
            hips_start = rough_hips_start
            for z in range(rough_hips_start, ribs_end, -1):
                view_l, view_r = getDataFractions(
                    bones[z, :, :],
                    fraction_defs=[frac_left, frac_right],
                    mask=fatlessbody[z, :, :],
                )
                if np.sum(view_l == 1) == 0 or np.sum(view_r == 1) == 0:
                    hips_start = z
                    break

        # remove anything thats between end of lungs and start of hip bones, is not spine, is not directly under surface (ribs).
        # - this should remove kidney stones, derbis in intestines and any high HU "sediments"
        for z in range(lungs_stats["max_area_z"], hips_start):
            view_spine = getDataFractions(
                bones[z, :, :], fraction_defs=[frac_spine], mask=fatlessbody[z, :, :]
            )
            tmp = view_spine.copy()
            bones[z, :, :][(bones[z, :, :] != 0) & (b_surface[z, :, :] == 0)] = 2
            view_spine[:, :] = tmp[:, :]  # TODO - test if changes original array
        # ed = sed3.sed3(data3d, seeds=bones); ed.show()
        # readd seed blobs that are connected to good seeds (half removed ribs in lower part of body)
        # as maybe bones (remove seeds)
        bones[
            (regionGrowing(bones != 0, bones == 1, bones != 0, mode="watershed") == 1)
            & (bones == 2)
        ] = 0

        ### Region growing - from seeds gained from high threshold, to mask gained by low threshold
        logger.debug("Region growing")
        bones_low = data3d > cls.BONES_THRESHOLD_LOW

        # parts that have both types of seeds should be removed for safety, if they have more bad seeds
        bones_low_label = skimage.measure.label(bones_low, background=0)
        for u in np.unique(bones_low_label)[1:]:
            good = np.sum(bones[bones_low_label == u] == 1)
            bad = np.sum(bones[bones_low_label == u] == 2)
            if bad > good:
                bones[(bones_low_label == u) & (bones != 0)] = 2

        # anything that is futher from seeds then BONES_LOW_MAX_DST is not bone
        bones_dst = scipy.ndimage.morphology.distance_transform_edt(
            bones != 1, sampling=spacing
        )
        bones[(bones_dst > cls.BONES_LOW_MAX_DST) & bones_low] = 2

        # use inverted data3d, so we can use 'watershed' as more then just basic region growing algorithm.
        # - bones become dark -> basins; tissues become lighter -> hills
        # ed = sed3.sed3(data3d, contour=bones_low, seeds=bones); ed.show()
        bones = (
            regionGrowing(
                skimage.util.invert(data3d), bones, bones_low, mode="watershed"
            )
            == 1
        )

        ### closing holes in segmented bones
        logger.debug("closing holes in segmented bones")
        bones = binaryClosing(bones, structure=getSphericalMask(5, spacing=spacing))
        bones = binaryFillHoles(bones, z_axis=True)

        # remove anything outside of fatless body
        bones[fatlessbody == 0] = 0

        # ed = sed3.sed3(data3d, contour=bones); ed.show()
        return bones

    DIAPHRAGM_SOBEL_THRESHOLD = -10
    DIAPHRAGM_MAX_LUNGS_END_DIST = (
        100
    )  # mm # TODO - maybe 150? (some data is cut too early)

    @classmethod
    def getDiaphragm(cls, data3d, spacing, lungs, body):  # TODO - improve
        """ Returns interpolated shape of Thoracic diaphragm (continues outsize of body) """
        logger.info("getDiaphragm()")
        if np.sum(lungs) == 0:
            logger.warning(
                "Couldn't find proper diaphragm, because we dont have lungs! Using a fake one that's in diaphragm[0,:,:]."
            )
            diaphragm = np.zeros(data3d.shape, dtype=np.bool).astype(np.bool)
            diaphragm[0, :, :] = 1
            diaphragm[body == 0] = 0
            return diaphragm

        # get edges of lungs on z axis
        diaphragm = (
            scipy.ndimage.filters.sobel(lungs.astype(np.int16), axis=0)
            < cls.DIAPHRAGM_SOBEL_THRESHOLD
        )

        # create diaphragm heightmap
        heightmap = np.zeros((diaphragm.shape[1], diaphragm.shape[2]), dtype=np.float)
        lungs_stop = lungs.shape[0] - getDataPadding(lungs)[0][1]
        diaphragm_start = max(
            0, lungs_stop - int(cls.DIAPHRAGM_MAX_LUNGS_END_DIST / spacing[0])
        )
        for y in range(diaphragm.shape[1]):
            for x in range(diaphragm.shape[2]):
                if np.sum(diaphragm[:, y, x]) == 0:
                    heightmap[y, x] = np.nan
                else:
                    tmp = diaphragm[:, y, x][::-1]
                    z = len(tmp) - np.argmax(tmp) - 1
                    if z < diaphragm_start:
                        # make sure that diaphragm is not higher then lowest lungs point -100mm
                        heightmap[y, x] = np.nan
                    else:
                        heightmap[y, x] = z

        # interpolate missing values
        height_median = np.nanmedian(heightmap)
        x = np.arange(0, heightmap.shape[1])
        y = np.arange(0, heightmap.shape[0])
        heightmap = np.ma.masked_invalid(heightmap)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~heightmap.mask]
        y1 = yy[~heightmap.mask]
        newarr = heightmap[~heightmap.mask]
        heightmap = scipy.interpolate.griddata(
            (x1, y1),
            newarr.ravel(),
            (xx, yy),
            method="linear",
            fill_value=height_median,
        )
        # ed = sed3.sed3(np.expand_dims(heightmap, axis=0)); ed.show()

        # 2D heightmap -> 3D diaphragm
        diaphragm = np.zeros(diaphragm.shape, dtype=np.bool).astype(np.bool)
        for y in range(diaphragm.shape[1]):
            for x in range(diaphragm.shape[2]):
                z = int(heightmap[y, x])
                diaphragm[: min(z + 1, diaphragm.shape[0]), y, x] = 1

        # make sure that diaphragm is lower then lungs volume
        diaphragm[lungs] = 1
        for y in range(diaphragm.shape[1]):
            for x in range(diaphragm.shape[2]):
                tmp = diaphragm[:, y, x][::-1]
                z = len(tmp) - np.argmax(tmp) - 1
                diaphragm[: min(z + 1, diaphragm.shape[0]), y, x] = 1

        # remove any data outside of body
        diaphragm[body == 0] = 0

        # ed = sed3.sed3(data3d, seeds=diaphragm); ed.show()
        return diaphragm

    # vessels threshold detection
    VESSELS_ROUGH_THRESHOLD = [70, 170]
    VESSELS_CANNY_MEDIAN = 10
    VESSELS_HOUGH_RADII = np.arange(
        5, 16, 1
    )  # in mm, radius of aorta is 10-14mm, venacava is 24mm
    VESSELS_HOUGH_THRESHOLD = 0.4
    VESSELS_HOUGH_INTENSITY_THRESHOLD = VESSELS_ROUGH_THRESHOLD[0] + 5
    VESSELS_HOUGH_RADII_DILATION = 10  # mm
    # vessels segmentation
    VESSELS_SPINE_BLOB_RADII = 20  # mm
    VESSELS_SEEDS_MIN_DIST = 15  # mm
    VESSELS_SEEDS_SURFACE_DIST = 40  # mm
    VESSELS_OBJ_MIN_SIZE = 10 ** 3  # mm3

    @classmethod
    def getVessels(
        cls, data3d, spacing, fatlessbody, bones, bones_stats, kidneys, threshold=None
    ):
        """
        Tabular value of blood radiodensity is 13-50 HU, but when contrast agent is used, it
        rises to roughly 100-240 HU. Threshold must be detected automaticaly.
        """
        logger.info("getVessels()")
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        voxel_avrg_2d_spacing = (spacing[1] + spacing[2]) / 2.0

        ##########################
        # Find Vessels Threshold #
        ##########################
        if threshold is None:
            logger.debug("Find Vessels Threshold")

            def get_normed_intensity_range(data, low, high):
                """ Converts <LOW,HIGH> to <0,255> """
                data = data.astype(np.float32)
                data[data < low] = low
                data[data > high] = high
                data -= low
                data *= 255.0 / (high - low)
                data = data.astype(np.uint8)
                return data

            # use rough threshold, convert range to <0;255> and remove small details from data to prepare for edge detection
            data3d_c = get_normed_intensity_range(
                data3d,
                low=cls.VESSELS_ROUGH_THRESHOLD[0],
                high=cls.VESSELS_ROUGH_THRESHOLD[1],
            )
            for z in range(data3d.shape[0]):
                data3d_c[z, :, :] = scipy.ndimage.filters.median_filter(
                    data3d_c[z, :, :], cls.VESSELS_CANNY_MEDIAN
                )
            # ed = sed3.sed3(data3d_c); ed.show()

            # find edges in data
            edges = np.zeros(data3d.shape, dtype=np.bool).astype(np.bool)
            for z in range(data3d.shape[0]):
                edges[z, :, :] = skimage.feature.canny(
                    data3d_c[z, :, :], sigma=0.0, low_threshold=150, high_threshold=200
                )

            # Detect radii
            hough_res = np.zeros(data3d.shape, dtype=np.float32)
            for z in range(data3d.shape[0]):
                hough_res[z, :, :] = np.max(
                    skimage.transform.hough_circle(
                        edges[z, :, :],
                        np.round(
                            cls.VESSELS_HOUGH_RADII.astype(np.float)
                            / voxel_avrg_2d_spacing
                        ).astype(np.int),
                    ),
                    axis=0,
                )
            # ed = sed3.sed3(hough_res, seeds=edges); ed.show()

            # remove unimportant detected radii
            for z in range(data3d.shape[0]):
                s_spine = getDataFractions(
                    hough_res[z, :, :],
                    fraction_defs=[{"h": (0.25, 1), "w": (0.40, 0.60)}],
                    mask=fatlessbody[z, :, :],
                    return_slices=True,
                )
                # remove everything not in spine section
                tmp = hough_res[z, :, :][s_spine].copy()
                hough_res[z, :, :] = np.zeros(
                    hough_res[z, :, :].shape, dtype=hough_res.dtype
                )
                hough_res[z, :, :][s_spine] = tmp
                # remove everything behind spine
                tmp = np.zeros(bones[z, :, :].shape, dtype=bones.dtype)
                tmp[s_spine] = bones[z, :, :][s_spine].copy()
                max_spine_y = np.min(firstNonzero(tmp, axis=0, invalid_val=np.inf))
                if max_spine_y != np.inf:
                    hough_res[z, int(max_spine_y) :, :] = 0
                # remove everything thats on voxel with lower denzity then threshold
                hough_res[
                    z, (data3d[z, :, :] < cls.VESSELS_HOUGH_INTENSITY_THRESHOLD)
                ] = 0

            # TODO - remove everything up of lungs_end-OFFSET -> current version works well only on abdomen only data

            # threshold radii
            hough_res = hough_res > cls.VESSELS_HOUGH_THRESHOLD
            # ed = sed3.sed3(data3d_c, seeds=hough_res); ed.show()

            # remove outliners and get pointsin vessels volume
            hough_res = scipy.ndimage.binary_dilation(
                hough_res,
                structure=getDiskMask(
                    cls.VESSELS_HOUGH_RADII_DILATION, spacing=spacing
                ),
            )
            hough_res_label = skimage.measure.label(hough_res, background=0)
            for u in np.unique(hough_res_label)[1:]:
                # remove if connected centers are not at more then N slices
                if (
                    hough_res.shape[0] - np.sum(getDataPadding(hough_res_label == u)[0])
                ) <= 5:
                    hough_res_label[hough_res_label == u] = 0
            hough_res = hough_res_label != 0
            hough_res[data3d < cls.VESSELS_HOUGH_INTENSITY_THRESHOLD] = 0
            # ed = sed3.sed3(data3d_c, seeds=hough_res); ed.show()

            # test if any vessels were detected at all
            if np.sum(hough_res) == 0:
                logger.warning(
                    "Could not detect vessels! (In vessels threshold detection)"
                )
                return np.zeros(data3d.shape, dtype=np.bool).astype(np.bool)

            # calculate statistics and thresholds
            d = data3d[hough_res]
            d_p = {}
            d_min = int(np.min(d))
            d_max = int(np.max(d))
            logger.debug("Vessels intensity min: %i, max: %i" % (d_min, d_max))
            for p in range(5, 100, 5):
                d_p[p] = int(np.percentile(d, p))
            logger.debug("Vessels intensity percentiles: %s" % str(sorted(d_p.items())))
            threshold = [d_p[10], d_p[90] + 50]
            logger.debug(
                "Detected vessels threshold: <%i,%i>" % (threshold[0], threshold[1])
            )

        ########################
        # Vessels Segmentation #
        ########################
        logger.debug("Vessels Segmentation")

        ## threshold vessels
        vessels = (data3d > threshold[0]) & (data3d < threshold[1])
        # ed = sed3.sed3(data3d, seeds=vessels); ed.show()
        vessels = binaryFillHoles(vessels, z_axis=True)
        # ed = sed3.sed3(data3d, seeds=vessels); ed.show()

        ## region growing - init seeds
        seeds = np.zeros(vessels.shape, dtype=np.int8)
        # bad seeds - already segmented parts and center of spine
        seeds[kidneys] = 1
        seeds[bones] = 1
        if len(bones_stats["spine"]) != 0:
            spine_zmin = bones_stats["spine"][0][0]
            spine_zmax = bones_stats["spine"][-1][0]
            for z in range(
                spine_zmin, spine_zmax + 1
            ):  # draw seeds elipse at spine center
                sc = bones_stats["spine"][z - spine_zmin]
                sc = (sc[1], sc[2])
                rr, cc = skimage.draw.ellipse(
                    sc[0],
                    sc[1],
                    int(cls.VESSELS_SPINE_BLOB_RADII / spacing[1]),
                    int(cls.VESSELS_SPINE_BLOB_RADII / spacing[2]),
                    shape=seeds[z, :, :].shape,
                )
                seeds[z, rr, cc] = 1
        # good seeds - min distance from already segmented parts
        fatlessbody_dst = scipy.ndimage.morphology.distance_transform_edt(
            fatlessbody, sampling=spacing
        )
        bad_seeds_dst = scipy.ndimage.morphology.distance_transform_edt(
            seeds != 1, sampling=spacing
        )
        seeds[
            (bad_seeds_dst > cls.VESSELS_SEEDS_MIN_DIST)
            & (fatlessbody_dst > cls.VESSELS_SEEDS_SURFACE_DIST)
        ] = 2
        seeds[(seeds == 2) & (vessels == 0)] = 0  # only in possible vessels volume

        ## region growing - run it
        # ed = sed3.sed3(data3d, seeds=seeds, contour=vessels); ed.show()
        vessels = regionGrowing(
            skimage.util.invert(data3d), seeds, vessels, mode="watershed"
        )
        # ed = sed3.sed3(data3d, seeds=seeds, contour=vessels); ed.show()
        vessels = vessels == 2

        ## remove small objects that are not connected to vessel tree
        vessels = skimage.morphology.remove_small_objects(
            vessels,
            min_size=int(cls.VESSELS_OBJ_MIN_SIZE / voxel_volume),
            connectivity=2,
        )
        # vessels = scipy.ndimage.binary_opening(vessels, structure=np.ones((3,3,3)))

        # TODO
        # remove all that are not connected to objects before and around spine

        # TODO - remove everything before heart? -> current version works well only on abdomen only data

        return vessels

    KIDNEYS_BINARY_OPENING = 10

    @classmethod
    def getKidneys(cls, data3d, spacing, cls_output, fatlessbody, diaphragm, liver):
        # TODO
        # - some data can have only one kidney
        # - potrebuje hodne vylepsit (zkus se podivat na predchozi verzi co nepouzivala patlas)
        logger.info("getKidneys()")
        # output of classifier
        data = cls_output
        # cleaning
        data[fatlessbody == 0] = 0
        data[diaphragm] = 0
        data[liver] = 0
        # binary opening, but return 1 only if there was 1 in orginal data
        data = data & scipy.ndimage.morphology.binary_opening(
            data,
            structure=getSphericalMask(cls.KIDNEYS_BINARY_OPENING, spacing=spacing),
        )
        # return only 2 biggest objects
        data = getBiggestObjects(data, 2)

        return data

    LIVER_BINARY_OPENING = 20

    @classmethod
    def getLiver(cls, data3d, spacing, cls_output, fatlessbody, diaphragm):
        logger.info("getLiver()")
        # output of classifier
        data = cls_output
        # cleaning
        data[fatlessbody == 0] = 0
        data[diaphragm] = 0
        # binary opening, but return 1 only if there was 1 in orginal data
        data = data & scipy.ndimage.morphology.binary_opening(
            data, structure=getSphericalMask(cls.LIVER_BINARY_OPENING, spacing=spacing)
        )
        # return only biggest object
        data = getBiggestObjects(data, 1)

        return data

    SPLEEN_BINARY_OPENING = 10

    @classmethod
    def getSpleen(cls, data3d, spacing, cls_output, fatlessbody, diaphragm):
        logger.info("getSpleen()")
        # output of classifier
        data = cls_output
        # cleaning
        data[fatlessbody == 0] = 0
        data[diaphragm] = 0
        # binary opening, but return 1 only if there was 1 in orginal data
        data = data & scipy.ndimage.morphology.binary_opening(
            data, structure=getSphericalMask(cls.SPLEEN_BINARY_OPENING, spacing=spacing)
        )
        # return only biggest object
        data = getBiggestObjects(data, 1)

        return data

    ##################
    ### Statistics ###
    ##################

    LUNGS_HULL_SYM_LIMIT = 0.1  # percent

    @classmethod
    def analyzeLungs(cls, lungs, spacing, fatlessbody):
        logger.info("analyzeLungs()")

        out = {
            "start": 0,
            "end": 0,  # start and end of lungs on z-axis
            "max_area_z": 0,  # idx of slice with biggest lungs area
        }
        if np.sum(lungs) == 0:
            logger.warning(
                "Since no lungs were found, defaulting start and end of lungs to 0, etc.."
            )
            return out

        lungs_pad = getDataPadding(lungs)
        out["start"] = lungs_pad[0][0]
        out["end"] = lungs.shape[0] - lungs_pad[0][1]
        out["max_area_z"] = np.argmax(np.sum(np.sum(lungs, axis=1), axis=1))

        return out

    @classmethod
    def analyzeBones(
        cls, bones, spacing, fatlessbody, lungs_stats
    ):  # TODO - clean, add ribs start/end (maybe)
        logger.info("analyzeBones()")

        # out = {"spine":[], "hips_joints":[], "hips_start":[]}

        # if np.sum(bones) == 0:
        #     logger.warning("Since no bones were found, returning empty values")
        #     return out

        # # merge near "bones" into big blobs
        # bones = binaryClosing(bones, structure=getSphericalMask(20, spacing=spacing)) # takes around 1m

        # # define sizes of spine and hip sections
        # frac_left = {"h":(0,1),"w":(0,0.40)}
        # frac_spine = {"h":(0.25,1),"w":(0.40,0.60)}
        # frac_front = {"h":(0,0.25),"w":(0.40,0.60)}
        # frac_right = {"h":(0,1),"w":(0.60,1)}

        # # get rough points
        # points_spine = []
        # for z in range(lungs_stats["start"], bones.shape[0]):
        #     view_left, view_spine, view_front, view_right = getDataFractions(bones[z,:,:], \
        #         fraction_defs=[frac_left,frac_spine,frac_front,frac_right], mask=fatlessbody[z,:,:])
        #     s_left, s_spine, s_front, s_right = getDataFractions(bones[z,:,:], \
        #         fraction_defs=[frac_left,frac_spine,frac_front,frac_right], mask=fatlessbody[z,:,:], \
        #         return_slices=True)

        #     # get volumes
        #     total_v = np.sum(bones[z,:,:])
        #     left_v = np.sum(view_left); spine_v = np.sum(view_spine); right_v = np.sum(view_right)

        #     # get centroids
        #     left_c = None; spine_c = None; right_c = None
        #     if left_v != 0:
        #         left_c = list(scipy.ndimage.measurements.center_of_mass(view_left))
        #         left_c[0] += s_left[0].start
        #         left_c[1] += s_left[1].start
        #     if spine_v != 0:
        #         spine_c = list(scipy.ndimage.measurements.center_of_mass(view_spine))
        #         spine_c[0] += s_spine[0].start
        #         spine_c[1] += s_spine[1].start
        #     if right_v != 0:
        #         right_c = list(scipy.ndimage.measurements.center_of_mass(view_right))
        #         right_c[0] += s_right[0].start
        #         right_c[1] += s_right[1].start

        #     # detect spine points
        #     if spine_v/total_v > 0.6:
        #         points_spine.append( (z, int(spine_c[0]), int(spine_c[1])) )

        #     # # try to detect hip joints
        #     # if (z >= lungs_end) and (left_v/total_v > 0.4) and (right_v/total_v > 0.4):
        #     #     # gets also leg bones
        #     #     #print(z, abs(left_c[1]-right_c[1]))
        #     #     if abs(left_c[1]-right_c[1]) < (180.0/spacing[2]): # max hip dist. 180mm
        #     #         # anything futher out should be only leg bones
        #     #         points_hips_joints_l.append( (z, int(left_c[0]), int(left_c[1])) )
        #     #         points_hips_joints_r.append( (z, int(right_c[0]), int(right_c[1])) )

        #     # # try to detect hip bones start on z axis
        #     # if (z >= lungs_end) and (left_v/total_v > 0.1):
        #     #     points_hips_start_l[z] = (z, int(left_c[0]), int(left_c[1]))
        #     # if (z >= lungs_end) and (right_v/total_v > 0.1):
        #     #     points_hips_start_r[z] = (z, int(right_c[0]), int(right_c[1]))

        #     sys.exit(0)

        # # fit curve to spine points and recalculate new points from curve
        # if len(points_spine) >= 2:
        #     points_spine = polyfit3D(points_spine)
        # out["spine"] = points_spine

        # return out
        ############################################################################################

        # remove every bone higher then lungs
        lungs_start = lungs_stats["start"]  # start of lungs on z-axis
        lungs_end = lungs_stats["end"]  # end of lungs on z-axis
        bones[:lungs_start, :, :] = 0  # definitely not spine or hips
        # remove front parts of ribs (to get correct spine center)
        for z in range(0, lungs_end):  # TODO - use getDataFractions
            bs = fatlessbody[z, :, :]
            pad = getDataPadding(bs)
            height = int(bones.shape[1] - (pad[1][0] + pad[1][1]))
            top_sep = pad[1][0] + int(height * 0.3)
            bones[z, :top_sep, :] = 0

        # merge near "bones" into big blobs
        bones[lungs_start:, :, :] = binaryClosing(
            bones[lungs_start:, :, :], structure=getSphericalMask(20, spacing=spacing)
        )  # takes around 1m

        # ed = sed3.sed3(data3d, contour=bones); ed.show()

        points_spine = []
        points_hips_joints_l = []
        points_hips_joints_r = []
        points_hips_start_l = {}
        points_hips_start_r = {}
        for z in range(lungs_start, bones.shape[0]):  # TODO - use getDataFractions
            # TODO - separate into more sections (spine should be only in middle-lower)
            bs = fatlessbody[z, :, :]
            # separate body/bones into 3 sections (on x-axis)
            pad = getDataPadding(bs)
            width = bs.shape[1] - (pad[1][0] + pad[1][1])
            left_sep = pad[1][0] + int(width * 0.35)
            right_sep = bs.shape[1] - (pad[1][1] + int(width * 0.35))
            left = bones[z, :, pad[1][0] : left_sep]
            center = bones[z, :, left_sep:right_sep]
            right = bones[z, :, right_sep : (bs.shape[1] - pad[1][1])]

            # calc centers and volumes
            left_v = np.sum(left)
            center_v = np.sum(center)
            right_v = np.sum(right)
            total_v = left_v + center_v + right_v
            if total_v == 0:
                continue
            left_c = [None, None]
            center_c = [None, None]
            right_c = [None, None]
            if left_v > 0:
                left_c = list(scipy.ndimage.measurements.center_of_mass(left))
                left_c[1] = left_c[1] + pad[1][0]
            if center_v > 0:
                center_c = list(scipy.ndimage.measurements.center_of_mass(center))
                center_c[1] = center_c[1] + left_sep
            if right_v > 0:
                right_c = list(scipy.ndimage.measurements.center_of_mass(right))
                right_c[1] = right_c[1] + right_sep

            # try to detect spine center
            if ((left_v / total_v < 0.2) or (right_v / total_v < 0.2)) and (
                center_v != 0
            ):
                points_spine.append((z, int(center_c[0]), int(center_c[1])))

            # try to detect hip joints
            if (
                (z >= lungs_end)
                and (left_v / total_v > 0.4)
                and (right_v / total_v > 0.4)
            ):
                # gets also leg bones
                # print(z, abs(left_c[1]-right_c[1]))
                if abs(left_c[1] - right_c[1]) < (
                    180.0 / spacing[2]
                ):  # max hip dist. 180mm
                    # anything futher out should be only leg bones
                    points_hips_joints_l.append((z, int(left_c[0]), int(left_c[1])))
                    points_hips_joints_r.append((z, int(right_c[0]), int(right_c[1])))

            # try to detect hip bones start on z axis
            if (z >= lungs_end) and (left_v / total_v > 0.1):
                points_hips_start_l[z] = (z, int(left_c[0]), int(left_c[1]))
            if (z >= lungs_end) and (right_v / total_v > 0.1):
                points_hips_start_r[z] = (z, int(right_c[0]), int(right_c[1]))

        # calculate centroid of hip points
        points_hips_joints = []
        if len(points_hips_joints_l) != 0:
            z, y, x = zip(*points_hips_joints_l)
            l = len(z)
            cl = (int(sum(z) / l), int(sum(y) / l), int(sum(x) / l))
            z, y, x = zip(*points_hips_joints_r)
            l = len(z)
            cr = (int(sum(z) / l), int(sum(y) / l), int(sum(x) / l))
            points_hips_joints = [cl, cr]

        # remove any spine points under detected hips
        if len(points_hips_joints) != 0:
            newp = []
            for p in points_spine:
                if p[0] < points_hips_joints[0][0]:
                    newp.append(p)
            points_spine = newp

        # fit curve to spine points and recalculate new points from curve
        if len(points_spine) >= 2:
            points_spine = polyfit3D(points_spine)

        # try to detect start of hip bones
        points_hips_start = [None, None]
        end_z = (
            bones.shape[0] - 1
            if len(points_hips_joints) == 0
            else points_hips_joints[0][0]
        )
        for z in range(end_z, lungs_start, -1):
            if z not in points_hips_start_l:
                if (z + 1) in points_hips_start_l:
                    points_hips_start[0] = points_hips_start_l[z + 1]
                break
        for z in range(end_z, lungs_start, -1):
            if z not in points_hips_start_r:
                if (z + 1) in points_hips_start_r:
                    points_hips_start[1] = points_hips_start_r[z + 1]
                break
        while None in points_hips_start:
            points_hips_start.remove(None)

        # seeds = np.zeros(bones.shape)
        # for p in points_spine_c: seeds[p[0], p[1], p[2]] = 2
        # for p in points_spine: seeds[p[0], p[1], p[2]] = 1
        # for p in points_hips_joints_l: seeds[p[0], p[1], p[2]] = 2
        # for p in points_hips_joints_r: seeds[p[0], p[1], p[2]] = 2
        # for p in points_hips_joints: seeds[p[0], p[1], p[2]] = 3
        # seeds = scipy.ndimage.morphology.grey_dilation(seeds, size=(1,5,5))
        # ed = sed3.sed3(data3d, contour=bones, seeds=seeds); ed.show()

        return {
            "spine": points_spine,
            "hips_joints": points_hips_joints,
            "hips_start": points_hips_start,
        }

    @classmethod
    def analyzeVessels(cls, data3d, spacing, vessels, bones_stats, liver):
        """ Returns: {"aorta":[], "vena_cava":[]} """
        logger.info("analyzeVessels()")
        out = {"aorta": [], "vena_cava": [], "liver": []}
        if np.sum(vessels) == 0:
            logger.warning("No vessels to find vessel points for!")
            return out

        ######################################
        # Detect where vessels go into liver #
        ######################################
        vessels_cut = scipy.ndimage.binary_opening(
            vessels, structure=np.ones((3, 3, 3))
        )
        # ed = sed3.sed3(data3d, contour=vessels, seeds=vessels_cut); ed.show()

        # get liver vessels # TODO - something eats crazy amount of memory here
        erosion_size = 30
        liver_cut = scipy.ndimage.morphology.binary_erosion(
            liver, structure=getSphericalMask(erosion_size, spacing=spacing)
        )
        liver_vessels = regionGrowing(
            vessels_cut,
            (vessels_cut & liver_cut),
            vessels_cut,
            spacing=spacing,
            max_dist=erosion_size,
            mode="watershed",
        )
        # ed = sed3.sed3(data3d, contour=vessels_cut, seeds=(liver_cut.astype(np.int8)+liver_vessels)); ed.show()

        # get connection borders
        connection = (
            scipy.ndimage.binary_dilation(liver_vessels, structure=np.ones((3, 3, 3)))
            - liver_vessels
        ) & vessels_cut
        # ed = sed3.sed3(data3d, contour=vessels_cut, seeds=connection); ed.show()

        # get connection points
        connection_l = skimage.measure.label(connection, background=0, connectivity=2)
        points = []
        for u in np.unique(connection_l)[1:]:
            points.append(scipy.ndimage.measurements.center_of_mass(connection_l == u))
        if len(points) != 0:
            # use only top and bottom point
            top = points[0]
            bottom = points[0]
            for p in points:
                if top[0] > p[0]:
                    top = p
                if bottom[0] < p[0]:
                    bottom = p
        out["liver"] = [top, bottom]

        ##############################
        # Detect Aorta and Vena Cava #
        ##############################
        points_spine = bones_stats["spine"]
        spine_zmin = points_spine[0][0]
        spine_zmax = points_spine[-1][0]
        rad = np.asarray([7, 8, 9, 10, 11, 12, 13, 14], dtype=np.float32)
        rad = list(rad / float((spacing[1] + spacing[2]) / 2.0))

        points_aorta = []
        points_vena_cava = []
        points_unknown = []
        for z in range(
            spine_zmin, spine_zmax + 1
        ):  # TODO - ignore space around heart (aorta), start under heart (vena cava)
            sc = points_spine[z - spine_zmin]
            sc = (sc[1], sc[2])
            vs = vessels[z, :, :]

            edge = skimage.feature.canny(vs, sigma=0.0)
            r = skimage.transform.hough_circle(edge, radius=rad) > 0.4
            r = np.sum(r, axis=0) != 0
            r[vs == 0] = 0  # remove centers outside segmented vessels
            r = scipy.ndimage.binary_closing(
                r, structure=np.ones((10, 10))
            )  # connect near centers

            # get circle centers
            if np.sum(r) == 0:
                continue
            rl = skimage.measure.label(r, background=0)
            centers = scipy.ndimage.measurements.center_of_mass(
                r, rl, range(1, np.max(rl) + 1)
            )

            # sort points between aorta, vena_cava and unknown
            for c in centers:
                c_zyx = (z, int(c[0]), int(c[1]))
                # spine center -> 100% aorta
                if sc[1] < c[1]:
                    points_aorta.append(c_zyx)
                # 100% venacava <- spine center - a bit more
                elif c[1] < (sc[1] - 20 / spacing[2]):
                    points_vena_cava.append(c_zyx)
                else:
                    points_unknown.append(c_zyx)

        # use watershed find where unknown points are
        cseeds = np.zeros(vessels.shape, dtype=np.int8)
        for p in points_aorta:
            cseeds[p[0], p[1], p[2]] = 1
        for p in points_vena_cava:
            cseeds[p[0], p[1], p[2]] = 2
        r = regionGrowing(vessels, cseeds, vessels, mode="watershed")
        # ed = sed3.sed3(data3d, contour=r, seeds=cseeds); ed.show()

        for p in points_unknown:
            if r[p[0], p[1], p[2]] == 1:
                points_aorta.append(p)
            elif r[p[0], p[1], p[2]] == 2:
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
                    sc = points_spine[z - spine_zmin]
                    dists = [((p[1] - sc[1]) ** 2 + (p[2] - sc[2]) ** 2) for p in pset]
                    points_aorta_new.append(pset[list(dists).index(min(dists))])
            points_aorta = points_aorta_new
        if len(points_vena_cava) >= 1:
            points_vena_cava_new = []
            for z, pset in groupby(points_vena_cava, key=itemgetter(0)):
                pset = list(pset)
                if len(pset) == 1:
                    points_vena_cava_new.append(pset[0])
                else:
                    sc = points_spine[z - spine_zmin]
                    dists = [((p[1] - sc[1]) ** 2 + (p[2] - sc[2]) ** 2) for p in pset]
                    points_vena_cava_new.append(pset[list(dists).index(min(dists))])
            points_vena_cava = points_vena_cava_new

        # polyfit curve
        if len(points_aorta) >= 2:
            points_aorta = polyfit3D(points_aorta)
        if len(points_vena_cava) >= 2:
            points_vena_cava = polyfit3D(points_vena_cava)

        out["aorta"] = points_aorta
        out["vena_cava"] = points_vena_cava

        return out
