#! /usr/bin/env python
# -*- coding: utf-8 -*-

from loguru import logger

# import logging
#
# logger = logging.getLogger(__name__)

import argparse

# import featurevector

# import apdb
#  apdb.set_trace();\
# import scipy.io
import numpy as np
import scipy
import scipy.ndimage
import skimage.measure
import skimage.transform
from . import chest_localization
import copy

# import sed3

# from imtools import misc, qmisc  # https://github.com/mjirik/imtools
# from imma.image import resize_to_mm, resize_to_shape

from .organ_detection_algo import OrganDetectionAlgo


def resize_to_mm(*args, **kwargs):
    import imma.image

    return imma.image.resize_to_mm(*args, **kwargs)


def resize_to_shape(*args, **kwargs):
    import imma.image

    return imma.image.resize_to_shape(*args, **kwargs)


class BodyNavigation:
    """ Range of values in input data must be <-1024;intmax> """

    def __init__(
        self,
        data3d,
        voxelsize_mm,
        use_new_get_lungs_setup=False,
        head_first=True,
        orientation_axcodes=None,
    ):
        # temporary fix for io3d <-512;511> value range bug
        if np.min(data3d) >= -512:
            data3d = data3d * 2
        # unresized data
        # self.data3d = data3d # some methods require original resolution
        self.orig_shape = data3d.shape
        self.voxelsize_mm = np.asarray(voxelsize_mm)

        # this drasticaly downscales data, but is faster
        self.working_vs = np.asarray([1.5] * 3)
        self._spine_filter_size_mm = np.array([100, 15, 15])
        self._spine_min_bone_voxels_ratio = 0.15
        self._spine_2nd_iter_dist_threshold_mm = 50
        self._bones_threshold_hu = 320
        self._cache_diaphragm_axial_i_vxsz = (
            None  # this will be caluclated if necessary
        )
        self.axcodes = orientation_axcodes if orientation_axcodes else "SPL"

        self._diaphragm_level_flat_area_proportion = 0.9  # check also local maxima
        self._lungs_max_density = -200
        self._diaphragm_level_min_dist_to_surface_mm = 1
        self._diaphragm_level_min_dist_to_sagittal_mm = 30
        self._symmetry_bones_threshold_hu = 430
        self._symmetry_gaussian_sigma_mm = 22.5
        self._symmetry_degrad_px = 5
        self._body_threshold = -300
        self._body_gaussian_sigma_mm = 3.

        if voxelsize_mm is None:
            self.data3dr = data3d
        else:
            self.data3dr = resize_to_mm(data3d, voxelsize_mm, self.working_vs)

        self.lungs = None
        self.spine = None
        self.body = None
        self.bones = None
        self.body_width = None
        self.body_height = None
        self.body_center_wvs = None
        self.diaphragm_mask = None
        self.angle = None
        self.spine_center_wvs = None
        self.ribs = None
        self.chest = None
        self.head_first = head_first
        self.use_new_get_lungs_setup = use_new_get_lungs_setup
        self.set_parameters()
        self.dist_sagittal = self.dist_to_sagittal
        self.dist_coronal = self.dist_to_coronal
        self.dist_axial = self.dist_to_axial
        self.body_center_wvs = None
        self.body_center_mm = None
        self.debug = False

    def set_parameters(self, version=1):
        if version == 0:
            self.GRADIENT_THRESHOLD = 12

        if version == 1:
            self.GRADIENT_THRESHOLD = 10

    def get_body(self, skip_resize=False):
        # create segmented 3d data
        sigma_px = self._body_gaussian_sigma_mm / np.mean(self.voxelsize_mm[1:])
        body = OrganDetectionAlgo.getBody(
            scipy.ndimage.filters.gaussian_filter(self.data3dr, sigma=sigma_px),
            self.working_vs, body_threshold=self._body_threshold
        )
        self.body = body

        # get body width and height
        widths = []
        heights = []
        for z in range(body.shape[0]):
            x_sum = np.sum(body[z, :, :], axis=0)  # suma kazdeho sloupcu
            x_start = next((i for i, x in enumerate(list(x_sum)) if x != 0), None)
            x_end = next(
                (i for i, x in enumerate(reversed(list(x_sum))) if x != 0), None
            )
            if x_start is None or x_end is None:
                width = 0
            else:
                width = (body.shape[2] - x_end) - x_start
            widths.append(width)

            y_sum = np.sum(body[z, :, :], axis=1)  # suma kazdeho radku
            y_start = next((i for i, y in enumerate(list(y_sum)) if y != 0), None)
            y_end = next(
                (i for i, y in enumerate(reversed(list(y_sum))) if y != 0), None
            )
            if y_start is None or y_end is None:
                height = 0
            else:
                height = (body.shape[1] - y_end) - y_start
            heights.append(height)

        # get value which is bigger then 90% of calculated values (to ignore bad data)
        body_width = np.percentile(np.asarray(widths), 90.0)
        body_height = np.percentile(np.asarray(heights), 90.0)
        # convert to original resolution
        body_width = body_width * (self.orig_shape[2] / float(self.body.shape[2]))
        body_height = body_height * (self.orig_shape[1] / float(self.body.shape[1]))
        # conver to mm
        self.body_width = body_width * float(self.voxelsize_mm[2])
        self.body_height = body_height * float(self.voxelsize_mm[1])

        self.body_center_wvs = np.mean(np.nonzero(body), 1)
        self.body_center_mm = self.body_center_wvs * self.working_vs

        if skip_resize:
            return self.body
        else:
            return resize_to_shape(self.body, self.orig_shape)

    def get_bones(self, skip_resize=False):
        """ Should have been named get_bones() """
        # prepare required data
        if self.body is None:
            self.get_body(skip_resize=True)  # self.body

        # filter out noise in data
        data3dr = scipy.ndimage.filters.median_filter(self.data3dr, 3)

        # tresholding
        # > 240 - still includes kidneys and small part of heart
        # > 280 - still a small bit of kidneys
        # > 350 - only edges of bones
        bones = data3dr > self._bones_threshold_hu
        del data3dr
        bones[self.body == 0] = 0  # cut out anything not inside body

        # close holes in data
        bones = scipy.ndimage.morphology.binary_closing(
            bones, structure=np.ones((3, 3, 3))
        )  # TODO - fix removal of data on edge slices
        self.bones = bones
        if skip_resize:
            return bones
        else:
            return resize_to_shape(bones, self.orig_shape)

    def get_spine(self, skip_resize=False):
        """
        Find voxels roughly representing spine. Use the binary bones,
        make filtration with with oblong object to remove smaller objects. Calculate XY center. Remove bones far from
        the center to remove hips and recalculate XY center again.
        :param skip_resize:
        :return:
        """
        # not used: Remove all woxels too far to remove hips and recalculate XY.
        # prepare required data
        if self.body is None:
            self.get_body(skip_resize=True)  # self.body
        if self.bones is None:
            self.get_bones(skip_resize=True)
        # dist_sag = self.dist_to_sagittal(return_in_working_voxelsize=True)

        # filter out noise in data
        spine_filter_size_px = self._spine_filter_size_mm / self.working_vs
        bones = self.bones.astype(float)
        # remove things far from sagittal to remove hips
        # bones[dist_sag > self._spine_max_dist_from_sagittal_mm] = 0
        # bones[dist_sag < -self._spine_max_dist_from_sagittal_mm] = 0
        bones[self.body == 0] = 0  # cut out anything not inside body
        data3dr = scipy.ndimage.filters.gaussian_filter(bones, spine_filter_size_px)

        # tresholding
        thr = min(0.9 * np.max(data3dr), self._spine_min_bone_voxels_ratio)
        spine = data3dr > thr
        import sed3
        logger.debug(f'thr={thr}')
        sed3.show_slices(data3dr, contour=spine)


        # compute temporary center
        spine_center_wvs = np.mean(np.nonzero(spine), 1)
        xx, yy = np.meshgrid(
            np.arange(0, data3dr.shape[1]), np.arange(0, data3dr.shape[2]), sparse=True
        )

        spine_mask_2d = np.sqrt(
            ((xx - spine_center_wvs[2]) * self.working_vs[2]) ** 2
            + ((yy - spine_center_wvs[1]) * self.working_vs[1]) ** 2
        )
        no_spine_mask_2d = spine_mask_2d > self._spine_2nd_iter_dist_threshold_mm
        # spine_mask_2d = np.zeros(data3dr.shape[1:])
        no_spine_mask = zcopy(no_spine_mask_2d, data3dr.shape)
        spine[no_spine_mask] = 0

        if self.debug:
            import matplotlib.pyplot as plt

            fig, axsss = plt.subplots(2, 2)
            axs = axsss.flatten()
            axs[0].imshow(np.max(data3dr, 0))
            axs[0].contour(
                np.max(
                    bones +
                    # (dist_sag > 0).astype(np.uint8) +
                    spine.astype(np.uint8) + no_spine_mask.astype(np.uint8),
                    0,
                )
            )
            axs[1].imshow(np.mean(data3dr, 0))
            axs[2].imshow(np.max(data3dr, 1))
            axs[3].imshow(np.mean(data3dr, 1))
            # axs[1].imshow(np.)
            plt.show()

        del bones
        del data3dr

        # compute center
        self.spine_center_wvs = np.mean(np.nonzero(spine), 1)

        # close holes in data
        # bones = scipy.ndimage.morphology.binary_closing(
        #     bones, structure=np.ones((3, 3, 3))
        # )  # TODO - fix removal of data on edge slices

        self.spine_center_mm = (
            self.spine_center_wvs
            * self.working_vs
            # / self.voxelsize_mm.astype(np.double)
        )
        self.spine_center_orig_px = (
            self.spine_center_wvs
            * self.working_vs
            / self.voxelsize_mm.astype(np.double)
        )

        # self.center2 = np.mean(np.nonzero(bones), 2)

        self.spine = spine
        if skip_resize:
            return spine
        else:
            return resize_to_shape(spine, self.orig_shape)

    # def get_coronal(self):
    #     if self.spine is None:
    #         self.get_spine(skip_resize=True)
    #     if self.angle is None:
    #         self.find_symmetry()
    #     spine_mean = np.mean(np.nonzero(self.spine), 1)
    #     rldst = np.ones(self.orig_shape, dtype=np.int16)
    def get_coronal(self, *args, **kwargs):
        return self.dist_to_coronal(*args, **kwargs)

    def get_lungs(self, skip_resize=False):
        self.diaphragm_mask = None
        if self.use_new_get_lungs_setup:
            return self.get_lungs_martin()
        else:
            return self.get_lungs_orig(skip_resize=skip_resize)

    def get_lungs_orig(self, skip_resize=False):
        lungs_density_threshold_hu = -150
        lungs = (
            scipy.ndimage.filters.gaussian_filter(self.data3dr, sigma=[4, 2, 2])
            > lungs_density_threshold_hu
        )

        if self.head_first:
            bottom_slice_id = -1
        else:
            bottom_slice_id = 0
        lungs[bottom_slice_id, :, :] = 1

        lungs = scipy.ndimage.morphology.binary_fill_holes(lungs)
        labs, n = scipy.ndimage.measurements.label(lungs == 0)
        cornerlab = [
            labs[0, 0, 0],
            labs[0, 0, -1],
            labs[0, -1, 0],
            labs[0, -1, -1],
            labs[-1, 0, 0],
            labs[-1, 0, -1],
            labs[-1, -1, 0],
            labs[-1, -1, -1],
        ]

        lb = np.median(cornerlab)
        labs[labs == lb] = 0

        labs[labs == labs[0, 0, 0]] = 0
        labs[labs == labs[0, 0, -1]] = 0
        labs[labs == labs[0, -1, 0]] = 0
        labs[labs == labs[0, -1, -1]] = 0
        labs[labs == labs[-1, 0, 0]] = 0
        labs[labs == labs[-1, 0, -1]] = 0
        labs[labs == labs[-1, -1, 0]] = 0
        labs[labs == labs[-1, -1, -1]] = 0

        lungs = labs > 0
        self.lungs = lungs
        # self.body = (labs == 80)
        if skip_resize:
            return self.lungs
        else:
            return resize_to_shape(lungs, self.orig_shape)

    def get_lungs_martin(self):
        """
        Improved version of get.lungs function. Call with ss.get_lungs if get_lungs_martin is default.
        first part checks if the CT-bed enables a clear segmentation of the background.
        Then we threshold and label all cavities. After that, we compare the mean value of the real grayscale-image in
        these labeled areas and only take the brighter ones. (Because in lung areas there are "white" alveoli)
        """

        lungs = (
            scipy.ndimage.filters.gaussian_filter(self.data3dr, sigma=[4, 2, 2]) > -450
        )
        lungs[0, :, :] = 1

        lungs = scipy.ndimage.morphology.binary_fill_holes(lungs)
        labs, n = scipy.ndimage.measurements.label(lungs == 0)
        cornerlab = [
            labs[0, 0, 0],
            labs[0, 0, -1],
            labs[0, -1, 0],
            labs[0, -1, -1],
            labs[-1, 0, 0],
            labs[-1, 0, -1],
            labs[-1, -1, 0],
            labs[-1, -1, -1],
        ]

        lb = np.median(cornerlab)
        labs[labs == lb] = 0

        labs[labs == labs[0, 0, 0]] = 0
        labs[labs == labs[0, 0, -1]] = 0
        labs[labs == labs[0, -1, 0]] = 0
        labs[labs == labs[0, -1, -1]] = 0
        labs[labs == labs[-1, 0, 0]] = 0
        labs[labs == labs[-1, 0, -1]] = 0
        labs[labs == labs[-1, -1, 0]] = 0
        labs[labs == labs[-1, -1, -1]] = 0

        lungs = labs > 0
        self.lungs = lungs
        # self.body = (labs == 80)

        # new code transformed into function

        segmented = self.data3dr < -400

        preselection_cavities = lungs * segmented

        # setting the first slice to one to close all of the cavities, so the fill holes works better
        first_slice = copy.copy(
            preselection_cavities[-1, :, :]
        )  # -1 means last slice, which is here the uppest in the image
        preselection_cavities[-1, :, :] = 1
        precav_filled = scipy.ndimage.morphology.binary_fill_holes(
            preselection_cavities
        )
        precav_filled[-1, :, :] = first_slice

        precav_erosion = scipy.ndimage.morphology.binary_erosion(precav_filled)
        labeled = skimage.morphology.label(precav_erosion)

        for f in range(1, np.max(labeled) + 1):

            cavity = self.data3dr[labeled == f]

            cavity_mean_intensity = np.std(cavity)

            if cavity_mean_intensity > 50:  # not too sure about the value of 50
                # idea would be to take the mean value of the highest ones and set the little lower one as the limit
                # this sets not lung-areas to zero. Theoretically :D
                precav_erosion[labeled == f] = 0
                # print(cavity_mean_intensity)

        precav_erosion = scipy.ndimage.morphology.binary_dilation(
            precav_filled
        )  # dilation becuase of erosion before

        # return precav_erosion
        self.lungs = precav_erosion
        return resize_to_shape(precav_erosion, self.orig_shape)

    def get_center_mm(self):
        ii = self.get_diaphragm_axial_position_index(return_in_working_voxelsize=True)
        # self.get_diaphragm_mask(skip_resize=True)
        self.get_spine(skip_resize=True)

        self.center = np.array(
            [
                # self.diaphragm_mask_level,
                ii,
                self.spine_center_wvs[0],
                self.spine_center_wvs[1],
            ]
        )
        self.center_mm = self.center * self.working_vs
        return self.center_mm

    def get_center(self):
        """
        Return center in orig pixels coordinates.
        :return:
        """
        self.get_center_mm()
        self.center_orig = (
            self.center * self.voxelsize_mm / self.working_vs.astype(np.double)
        )

        return self.center_orig

    def get_chest(self):
        """ Compute, where is the chest in CT data.
            :return: binary array
        """

        if self.chest is None:
            self.get_ribs()

        return self.chest

    def get_ribs(self):
        """ Compute, where are the ribs in CT data.
            :return: binary array
        """
        if self.body is None:
            self.get_body(skip_resize=True)
        if self.lungs is None:
            self.get_lungs(skip_resize=True)

        chloc = chest_localization.ChestLocalization(
            bona_object=self, data3dr_tmp=self.data3dr
        )

        body = chloc.clear_body(self.body)
        coronal = self.dist_to_coronal(return_in_working_voxelsize=True)

        final_area_filter = chloc.area_filter(self.data3dr, body, self.lungs, coronal)
        location_filter = chloc.dist_hull(final_area_filter)
        intensity_filter = chloc.strict_intensity_filter(self.data3dr)
        deep_filter = chloc.deep_struct_filter_old(
            self.data3dr
        )  # potrebuje upravit jeste

        ribs = (
            intensity_filter & location_filter & final_area_filter & body & deep_filter
        )

        # ribs_sum = intensity_filter.astype(float) + location_filter.astype(float) + final_area_filter.astype(float) + deep_filter.astype(float)

        # oriznuti zeber (a take hrudniku) v ose z
        z_border = chloc.process_z_axe(ribs, self.lungs, "001")
        ribs[0:z_border, :, :] = False
        final_area_filter[0:z_border, :, :] = False

        # chloc.print_it_all(ss, data3dr_tmp, final_area_filter*2, pattern+"area")
        # chloc.print_it_all(self, self.data3dr, ribs*2, pattern+"thr")
        # chloc.print_it_all(self, self.data3dr>220, ribs*3, pattern)

        # zebra
        self.ribs = ribs
        # hrudnik
        self.chest = final_area_filter

        return ribs

    def _distance_transform(self, data):
        return scipy.ndimage.morphology.distance_transform_edt(
            data, sampling=self.voxelsize_mm
        )

    def _resize_to_orig_shape(self, data):
        return resize_to_shape(data, self.orig_shape)

    def _resize_and_dist(self, data):
        return self._distance_transform(self._resize_to_orig_shape(data))

    def dist_to_chest(self):
        """
        Get distance in mm.
        :return:
        """
        if self.chest is None:
            self.get_ribs()
        # chest = self._resize_to_orig_shape(self.chest)
        ld_positive = scipy.ndimage.morphology.distance_transform_edt(self.chest)
        ld_negative = scipy.ndimage.morphology.distance_transform_edt(1 - self.chest)
        ld = ld_positive - ld_negative
        ld = ld * float(self.working_vs[0])  # convert distances to mm
        return self._resize_to_orig_shape(ld)

    def dist_to_ribs(self):
        """
        Distance to ribs.
        The distance is grater than zero inside of body and outside of body
        """
        if self.ribs is None:
            self.get_ribs()
        ld = scipy.ndimage.morphology.distance_transform_edt(1 - self.ribs)
        ld = ld * float(self.working_vs[0])  # convert distances to mm
        return resize_to_shape(ld, self.orig_shape)
        # return self._resize_and_dist(1 - self.ribs)

    def dist_to_surface(self, return_in_working_voxelsize=False):
        """
        Positive values in mm inside of body.
        :return:
        """
        if self.body is None:
            self.get_body(skip_resize=True)
        # return self._resize_and_dist(self.body)
        ld = scipy.ndimage.morphology.distance_transform_edt(self.body)
        ld = ld * float(self.working_vs[0])  # convert distances to mm
        if return_in_working_voxelsize:
            return ld
        else:
            return resize_to_shape(ld, self.orig_shape, mode="mirror")

    def dist_to_lungs(self):
        if self.lungs is None:
            self.get_lungs()
        ld = scipy.ndimage.morphology.distance_transform_edt(1 - self.lungs)
        ld = ld * float(self.working_vs[0])  # convert distances to mm
        return resize_to_shape(ld, self.orig_shape)

    def dist_to_spine(self):
        if self.spine is None:
            self.get_spine(skip_resize=True)
        # ld = scipy.ndimage.morphology.distance_transform_edt(1 - self.spine)
        # ld = ld * float(self.working_vs[0])  # convert distances to mm

        # working vs
        # shape = self.data3dr.shape
        # vs = self.working_vs
        # center = self.spine_center_wvs

        # orig size
        shape = self.orig_shape
        vs = self.voxelsize_mm
        center = self.spine_center_mm
        # spine_center_wvs = np.mean(np.nonzero(spine), 1)
        xx, yy = np.meshgrid(
            np.arange(0, shape[1]) * vs[1], np.arange(0, shape[2]) * vs[2], sparse=True
        )

        spine_dist_2d = np.sqrt(((xx - center[2])) ** 2 + ((yy - center[1])) ** 2)
        # spine_mask_2d = np.zeros(data3dr.shape[1:])
        spine_dist_3d = zcopy(spine_dist_2d, shape)
        # return resize_to_shape(ld, self.orig_shape)
        return spine_dist_3d

    def find_symmetry(self, return_img=False):
        if self.spine is None:
            self.get_spine()
        if self.body is None:
            self.get_body()

        degrad = self._symmetry_degrad_px
        vector = self.spine_center_wvs[1:] - self.body_center_wvs[1:]

        img = np.sum(self.data3dr > self._symmetry_bones_threshold_hu, axis=0)

        if self.debug:
            import matplotlib.pyplot as plt

            plt.imshow(img)
            plt.colorbar()
            plt.show()
        init_angle = 90 - np.degrees(np.arctan2(vector[0], vector[1]))
        init_point = self.body_center_wvs[1:]
        sigma_px = self._symmetry_gaussian_sigma_mm / np.mean(self.working_vs[1:])
        # from body_center to spine
        tr0, tr1, angle = find_symmetry(
            img, degrad, debug=self.debug, init_angle=init_angle, init_point=init_point, sigma=sigma_px
        )
        self.angle = angle
        self.symmetry_point_wvs = np.array([tr0, tr1])
        # self.symmetry_point_mm = self.symmetry_point_wvs * self.working_vs[:2]
        # self.symmetry_point_mm = self.symmetry_point_wvs * self.voxelsize_mm[:2]
        if return_img:
            return img

    def dist_to_sagittal(self, return_in_working_voxelsize=False):
        if self.angle is None:
            self.find_symmetry()
        if self.body is None:
            self.get_body()
        if self.spine is None:
            self.get_spine()

        if return_in_working_voxelsize:
            symmetry_point = np.asarray(self.symmetry_point_wvs)
            shape = self.data3dr.shape
            vs0 = self.working_vs[1]
        else:
            symmetry_point_orig_res = (
                self.symmetry_point_wvs
                * self.working_vs[1:]
                / self.voxelsize_mm[1:].astype(np.double)
            )
            symmetry_point = symmetry_point_orig_res
            shape = self.orig_shape
            vs0 = self.voxelsize_mm[1]

        # from body_center to spine
        vector = self.spine_center_wvs[1:] - self.body_center_wvs[1:]

        right_vector = np.asarray([vector[1], -vector[0]])
        point_on_right = symmetry_point + right_vector

        z, sgn = split_with_line(
            symmetry_point,
            self.angle,
            shape[1:],
            voxelsize_0=vs0,
            point_in_positive_halfplane=point_on_right,
            return_sgn=True,
        )
        rldst = zcopy(z, shape, dtype=np.int16)
        if sgn < 0:
            self.angle = (self.angle + 180) % 360  # maybe 180

        return rldst

    def get_diaphragm_axial_position_index(
        self, return_in_working_voxelsize=False, return_mask=False, return_areas=False
    ):
        """

        :param return_in_working_voxelsize: return position index in working voxelsize
        :param return_mask:  mask in working voxelsize
        :param return_areas: Areas of lungs (without a middle around sagittal plane) in mm^2 for all slices
        :return:
        """
        if self._cache_diaphragm_axial_i_vxsz and not return_mask:
            ii = self._cache_diaphragm_axial_i_vxsz
            if not return_in_working_voxelsize:
                ii = ii * self.orig_shape[0] / self.data3dr.shape[0]
            return ii
        if self.spine is None:
            self.get_spine(skip_resize=True)
        if self.angle is None:
            self.find_symmetry()

        # self = bodynavigation.body_navigation.BodyNavigation(datap.data3d, datap.voxelsize_mm)
        dst_surf = self.dist_to_surface(return_in_working_voxelsize=True)
        dst_sagi = self.dist_sagittal(return_in_working_voxelsize=True)
        #     dst_coro = bn.dist_coronal()
        mns = []

        maska = dst_surf > self._diaphragm_level_min_dist_to_surface_mm
        maskb = np.abs(dst_sagi) > self._diaphragm_level_min_dist_to_sagittal_mm
        mask = maska & maskb
        mask_val = (mask & (self.data3dr < self._lungs_max_density)).astype(np.int8)

        #     maskc = mask & (dst_coro < thr_dist_to_coronal_mm)
        #     mask_val_c = (maskc & (datap.data3d < max_density)).astype(np.int8)

        #     voxel_axial_surface_mm2 = datap.voxelsize_mm[1] * datap.voxelsize_mm[2]
        voxel_axial_surface_mm2 = self.working_vs[1] * self.working_vs[2]
        # print(mask.shape, np.unique(mask), scipy.stats.describe(dst_surf.flatten()),
        #       scipy.stats.describe(dst_sagi.flatten()))
        for i in range(0, self.data3dr.shape[0]):
            mn = np.sum(mask_val[i, :, :])
            mns.append(mn * voxel_axial_surface_mm2)

        #         mnc = np.sum(mask_val_c[i,:,:])
        #         mnsc.append(mnc * voxel_axial_surface_mm2)
        ii_max = np.nanargmax(mns)

        # concider points of local extrema
        ids = scipy.signal.argrelextrema(np.asarray(mns), np.greater)

        # add to the absolute maximum to concidered points
        ids = set(ids[0])
        ids.add(ii_max)
        ids = np.asarray(list(ids))
        mnsa = np.asarray(mns)

        # take just indexes with value above some intensity level
        ids2 = ids[
            mnsa[ids] > (self._diaphragm_level_flat_area_proportion * mns[ii_max])
        ]
        if self.axcodes[0] == "S":
            ii = np.max(ids2)  # if axcode is "SPL"
        elif self.axcodes[0] == "I":
            ii = np.min(ids2)  # if axcode is "IPL"
        else:
            raise ValueError(f"Unsupported orientation_axcodes {self.axcodes}")

        self._cache_diaphragm_axial_i_vxsz = ii

        if not return_in_working_voxelsize:
            ii = ii * self.orig_shape[0] / self.data3dr.shape[0]

        out = [ii]
        if return_mask:
            out.append(mask)
        if return_areas:
            out.append(mnsa)
        # return ii
        return out[0] if len(out) == 1 else tuple(out)
        # ----------------

    def dist_to_diaphragm_axial(self, return_in_working_voxelsize=False):
        ii = self.get_diaphragm_axial_position_index(return_in_working_voxelsize)

        if return_in_working_voxelsize:
            output_shape = self.data3dr.shape
            voxelsize_0 = self.working_vs[0]
        else:
            output_shape = self.orig_shape
            voxelsize_0 = self.voxelsize_mm[0]

        height_mm = voxelsize_0 * output_shape[0]

        data = np.ones(output_shape)
        mul = np.linspace(0, height_mm, output_shape[0]).reshape(
            [output_shape[0], 1, 1]
        )
        mul = mul - (ii * voxelsize_0)
        return data * mul

    def dist_to_coronal(self, return_in_working_voxelsize=False):
        if self.spine is None:
            self.get_spine(skip_resize=True)
        if self.angle is None:
            self.find_symmetry()

        if return_in_working_voxelsize:
            shape = self.data3dr.shape
            spine_center = self.spine_center_wvs
        else:
            shape = self.orig_shape
            spine_center = self.spine_center_orig_px
        rldst = np.ones(shape, dtype=np.int16)

        point_in_positive_halfplane = self.body_center_wvs[1:]
        z = split_with_line(
            spine_center[1:],
            self.angle + 90,
            shape[1:],
            voxelsize_0=self.working_vs[1],
            point_in_positive_halfplane=point_in_positive_halfplane,
        )
        z = z * self.working_vs[1]
        for i in range(self.orig_shape[0]):
            rldst[i, :, :] = z

        return rldst

    def dist_to_axial(self):
        if self.diaphragm_mask is None:
            self.get_diaphragm_mask(skip_resize=True)
        axdst = np.ones(self.data3dr.shape, dtype=np.int16)
        axdst[0, :, :] = 0
        iz, ix, iy = np.nonzero(self.diaphragm_mask)
        # print 'dia level ', self.diaphragm_mask_level

        axdst = scipy.ndimage.morphology.distance_transform_edt(axdst) - int(
            self.diaphragm_mask_level
        )
        axdst = axdst * self.working_vs[0]
        return resize_to_shape(axdst, self.orig_shape)

    def dist_to_diaphragm(self):
        """
        Get positive values in superior to (above) diaphragm and negative values inferior (under) diaphragm.
        :return:
        """
        if self.diaphragm_mask is None:
            self.get_diaphragm_mask(skip_resize=True)
        dst = scipy.ndimage.morphology.distance_transform_edt(
            self.diaphragm_mask
        ) - scipy.ndimage.morphology.distance_transform_edt(  # , sampling=self.voxelsize_mm)
            1 - self.diaphragm_mask
        )  # , sampling=self.voxelsize_mm)

        dst = dst * self.working_vs[0]
        return resize_to_shape(dst, self.orig_shape)

    def _get_ia_ib_ic(self, axis):
        """
        according to axis gives order of of three dimensions
        :param axis: 0, 1 or 2 is allowed
        :return:
        """
        if axis == 0:
            ia = 0
            ib = 1
            ic = 2
        elif axis == 1:
            ia = 1
            ib = 0
            ic = 2
        elif axis == 2:
            ia = 2
            ib = 0
            ic = 1
        else:
            logger.error("Unrecognized axis")

        return ia, ib, ic

    def _filter_diaphragm_profile_image_remove_outlayers(
        self, profile, axis=0, tolerance=80
    ):
        # import bottleneck
        # tolerance * 1.5mm
        non_nan_profile = profile[~np.isnan(profile)]
        positive_non_nan_profile_values = non_nan_profile[non_nan_profile > 0]
        med = np.median(positive_non_nan_profile_values)

        profile[
            np.greater(np.abs(profile - med), tolerance, where=~np.isnan(profile))
        ] = None
        return profile

    def get_diaphragm_profile_image_with_empty_areas(
        self, axis=0, return_gradient_image=False
    ):
        """

        :param axis:
        :param return_gradient_image: If true, return profile_image, gradient_image
        :return: profile_image or (profile_image, gradient_image)
        """
        if self.lungs is None:
            self.get_lungs(skip_resize=True)
        if self.spine is None:
            self.get_spine(skip_resize=True)
        if self.angle is None:
            self.find_symmetry()
        if self.body is None:
            self.get_body(skip_resize=True)

        data = self.lungs
        ia, ib, ic = self._get_ia_ib_ic(axis)

        # gradient
        gr = scipy.ndimage.filters.sobel(data.astype(np.int16), axis=ia)
        # seg = (np.abs(ss.dist_coronal()) > 20).astype(np.uint8) + (np.abs(ss.dist_sagittal()) > 20).astype(np.uint8)
        if self.head_first:
            grt = gr < -self.GRADIENT_THRESHOLD
        else:
            grt = gr > self.GRADIENT_THRESHOLD

        # TODO zahodit velke gradienty na okraji tela,

        flat = self.nonzero_projection(grt, axis)

        # seg = (np.abs(self.dist_sagittal()) > zero_stripe_width).astype(np.uint8)
        # grt = grt * seg
        flat[flat == 0] = np.NaN
        if return_gradient_image:
            return flat, gr
        return flat

    def nonzero_projection(self, grt, axis):
        # nalezneme nenulove body
        ia, ib, ic = self._get_ia_ib_ic(axis)
        nz = np.nonzero(grt)

        # udelame z 3d matice placku, kde jsou nuly tam, kde je nic a jinde jsou
        # z-tove souradnice
        flat = np.zeros([grt.shape[ib], grt.shape[ic]])
        flat[(nz[ib], nz[ic])] = [nz[ia]]

        # symmetry_point_orig_res = self.symmetry_point * self.working_vs[1:] / self.voxelsize_mm[1:].astype(np.double)
        # odstranime z dat pruh kolem srdce. Tam byva obcas jicen, nebo je tam oblast nad srdcem
        # symmetry_point_pixels = self.symmetry_point/ self.working_vs[1:]
        return flat

    def filter_remove_outlayers(self, flat, minimum_value=0):
        """
        Remove outlayers using ellicptic envelope from scikits learn
        :param flat:
        :param minimum_value:
        :return:
        """
        from sklearn.covariance import EllipticEnvelope

        flat0 = flat.copy()
        flat0[np.isnan(flat)] = 0
        x, y = np.nonzero(flat0)
        # print np.prod(flat.shape)
        # print len(y)

        z = flat[(x, y)]

        data = np.asarray([x, y, z]).T

        clf = EllipticEnvelope(contamination=0.1)
        clf.fit(data)
        y_pred = clf.decision_function(data)

        out_inds = y_pred < minimum_value
        flat[(x[out_inds], y[out_inds])] = np.NaN
        return flat

    def filter_ignoring_nan(self, flat, kernel_size_mm=None, max_dist_mm=30):
        """
        Compute filtered plane and removes pixels wiht distance grater then max_dist_mm

        :param flat:
        :param kernel_size_mm:
        :param max_dist_mm:
        :return:
        """

        if kernel_size_mm is None:
            kernel_size_mm = [150, 150]

        # kernel_size must be odd - lichý
        kernel_size = np.asarray(kernel_size_mm) / self.working_vs[1:]
        # print 'ks1 ', kernel_size
        odd = kernel_size % 2
        kernel_size = kernel_size + 1 - odd
        # print 'ks2 ', kernel_size

        # metoda 1
        kernel = np.ones(np.round(kernel_size).astype(np.int))
        kernel = kernel / (1.0 * np.prod(kernel_size))
        # flat = scipy.ndimage.filters.convolve(flat, kernel)

        # metoda 2
        # # flat = flat.reshape([flat.shape[0], flat.shape[1], 1])

        # Proper treatment of NaN values(ignoring them during convolution and replacing NaN pixels with interpolated values)
        import astropy.convolution

        flat_out = astropy.convolution.convolve(flat, kernel, boundary="extend")

        too_bad_pixels = np.greater(
            np.abs(flat_out - flat),
            (max_dist_mm / self.working_vs[0]),
            where=~np.isnan(flat),
        )

        flat[too_bad_pixels] = np.NaN
        # metoda 3
        # doplnime na nenulova mista střední hodnotu
        # flat_mask = np.isnan(flat)
        #
        # mn = np.mean(flat[flat_mask == False])
        #
        # flat_copy = flat.copy()
        # flat_copy[flat_mask] = mn
        #
        # flat_copy = scipy.ndimage.filters.gaussian_filter(flat_copy, sigma=sigma)
        # flat = flat_copy
        return flat

    def get_diaphragm_profile_image_orig_shape_mm(
        self, axis=0, preprocessing=True, return_preprocessed_image=False
    ):
        diaphragm_profile = self.get_diaphragm_profile_image(
            axis=axis,
            preprocessing=preprocessing,
            return_preprocessed_image=return_preprocessed_image,
        )
        diaphragm_profile_orig_shape = skimage.transform.resize(
            diaphragm_profile, self.orig_shape[1:]
        )
        diaphragm_profile_orig_shape_mm = (
            diaphragm_profile_orig_shape * self.working_vs[0]
        )
        return diaphragm_profile_orig_shape_mm

    def get_diaphragm_profile_image(
        self, axis=0, preprocessing=True, return_preprocessed_image=False
    ):
        """
        Diaphragm profile in pixels. Use get_diaphragm_profile_image_orig_shape_in_mm to get data in mm.
        :param axis:
        :param preprocessing:
        :param return_preprocessed_image:
        :return:
        """
        flat = self.get_diaphragm_profile_image_with_empty_areas(axis)

        if preprocessing:
            flat = self.remove_pizza(flat)
            flat = self._filter_diaphragm_profile_image_remove_outlayers(flat)
            # flat = self.filter_ignoring_nan(flat)
            # flat = self.filter_remove_outlayers(flat)

        # jeste filtrujeme ne jen podle stredni hodnoty vysky, ale i v prostoru
        # flat0 = flat==0
        # flat[flat0] = None
        #
        # flat = scipy.ndimage.filters.median_filter(flat, size=(40,40))
        # flat[flat0] = 0

        # something like interpolation
        # doplnime praznda mista v ploche mape podle nejblizsi oblasi
        # filter with big mask
        ou = fill_nan_with_nearest(flat.copy())
        ou = scipy.ndimage.filters.gaussian_filter(ou, sigma=5)

        # get back valid values on its places
        valid_flat_inds = 1 - np.isnan(flat)
        ou[valid_flat_inds] = flat[valid_flat_inds]
        # flat = ou

        ou = fill_nan_with_nearest(ou)
        # overal filter
        ou = scipy.ndimage.filters.median_filter(ou, size=(5, 5))
        ou = scipy.ndimage.filters.gaussian_filter(ou, sigma=3)

        # ou = self.__filter_diaphragm_profile_image(ou, axis)
        retval = [ou]
        if return_preprocessed_image:
            retval.append(flat)

        if len(retval) == 1:
            return retval[0]
        else:
            return tuple(retval)

    def __filter_diaphragm_profile_image(self, profile, axis=0):
        """
        filter do not go down in compare to pixel near to the back
        :param profile:
        :param axis:
        :return:
        """
        if axis == 0:

            profile_w = profile.copy()

            # profile_out = np.zeros(profile.shape)
            for i in range(profile_w.shape[0] - 1, 0, -1):
                profile_line_0 = profile_w[i, :]
                profile_line_1 = profile_w[i - 1, :]
                where_is_bigger = profile_line_1 < (profile_line_0 - 0)
                #     profile_line_out[where_is_bigger] = profile_line_0[where_is_bigger]
                profile_w[i - 1, where_is_bigger] = profile_line_0[where_is_bigger]
                profile_w[i - 1, np.negative(where_is_bigger)] = profile_line_1[
                    np.negative(where_is_bigger)
                ]
            #     profile_out[where_is_bigger, :] = profile_line_1
        else:
            logger.error("other axis not implemented yet")

        return profile_w
        # plt.imshow(profile_w, cmap='jet')

    def get_diaphragm_mask(self, axis=0, skip_resize=False):
        """
        Get False in inferior to diaphragm and True in superior to diaphragm.
        :param axis:
        :param skip_resize:
        :return: Boolean 3D ndarray
        """
        if self.lungs is None:
            self.get_lungs()
        ia, ib, ic = self._get_ia_ib_ic(axis)
        data = self.lungs
        ou = self.get_diaphragm_profile_image(axis=axis)
        # reconstruction mask array
        mask = np.zeros(data.shape)
        for i in range(mask.shape[ia]):
            if ia == 0:
                mask[i, :, :] = ou > i
            elif ia == 1:
                mask[:, i, :] = ou > i
            elif ia == 2:
                mask[:, :, i] = ou > i

        if not self.head_first:
            # invert mask
            mask = mask == False

        self.diaphragm_mask = mask

        # maximal point is used for axial ze
        # ro plane
        self.diaphragm_mask_level = np.median(ou)
        self.center0 = self.diaphragm_mask_level * self.working_vs[0]

        if skip_resize:
            return self.diaphragm_mask
        return resize_to_shape(self.diaphragm_mask, self.orig_shape)

    def remove_pizza(self, flat, zero_stripe_width=10, alpha0=-20, alpha1=40):
        """
        Remove circular sector from the image with center in spine
        :param flat: input 2D image
        :param zero_stripe_width: offset to pivot point. Pizza line is zero_stripe_width far
        :param alpha0: Additional start angle relative to detected orientation
        :param alpha1: Additional end angle relative to detected orientation
        :return:
        """
        spine_mean = np.mean(np.nonzero(self.spine), 1)
        spine_mean = spine_mean[1:]

        z1 = split_with_line(spine_mean, self.angle + alpha1, flat.shape)
        z2 = split_with_line(spine_mean, self.angle + alpha0, flat.shape)

        z1 = (z1 > zero_stripe_width).astype(np.int8)
        z2 = (z2 < -zero_stripe_width).astype(np.int8)
        # seg = (np.abs(z) > zero_stripe_width).astype(np.int)
        seg = z1 * z2

        flat[seg > 0] = np.NaN  # + seg*10
        # print 'sp ', spine_mean
        # print 'sporig ', symmetry_point_orig_res
        # flat = seg

        return flat


def prepare_images_for_symmetry_analysis(imin0, pivot):
    imin1 = imin0[:, ::-1]
    img = imin0

    padX0 = [img.shape[1] - pivot[1], pivot[1]]
    padY0 = [img.shape[0] - pivot[0], pivot[0]]
    imgP0 = np.pad(img, [padY0, padX0], "constant")

    img = imin1
    padX1 = [pivot[1], img.shape[1] - pivot[1]]
    padY1 = [img.shape[0] - pivot[0], pivot[0]]
    imgP1 = np.pad(img, [padY1, padX1], "constant")
    return imgP0, imgP1


def find_symmetry_parameters(
    imin0, trax, tray, angles, debug=False, return_criterium_min=False
):
    """

    :param imin0:
    :param trax:
    :param tray:
    :param angles:
    :param debug:
    :return:
    """
    vals = np.zeros([len(trax), len(tray), len(angles)])
    #     angles_vals = []
    minval = np.inf
    minimg = None
    minim0 = None
    minim1 = None
    # mindif = None
    for i, x in enumerate(trax):
        for j, y in enumerate(tray):
            try:
                pivot = [x, y]
                imgP0, imgP1 = prepare_images_for_symmetry_analysis(imin0, pivot)

                #             print 'y ', y
                for k, angle in enumerate(angles):

                    imr = scipy.ndimage.rotate(imgP1, angle, reshape=False)
                    dif = (imgP0 - imr) ** 2
                    sm = np.sum(dif)
                    vals[i, j, k] = sm
                    if debug:
                        if sm < minval:
                            minval = sm
                            mindif = dif
                            minim0 = imgP0
                            minim1 = imr
                            # minimg =
            except Exception as e:
                vals[i, j, :] = np.inf
    #             angles_vals.append(sm)

    # am = np.argmin(angles_vals)
    am = np.unravel_index(np.argmin(vals), vals.shape)
    # print am, ' min ', np.min(vals)
    if debug:

        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(1, 3, sharey=True)
        axs = axs.flatten()
        axs[0].imshow(mindif)
        axs[1].imshow(minim0)
        axs[2].imshow(minim1)
        fig.suptitle(f"angle={angles[am[2]]}, point=[{trax[am[0]]}, {tray[am[1]]}]")
        plt.show()

    if return_criterium_min:
        return trax[am[0]], tray[am[1]], angles[am[2]], np.min(vals)
    else:
        return trax[am[0]], tray[am[1]], angles[am[2]]


def find_symmetry(img, degrad=5, debug=False, sigma=15, init_angle=0, init_point=None):
    """

    :param img:
    :param degrad: resample to lower resolution by degrad
    :param debug:
    :param sigma: gaussian filter in pixels of input image. The degrad parameter is conciderd internally.
    :param init_angle:
    :param init_point:
    :return:
    """
    # imin0r = scipy.misc.pilutil.imresize(img, (np.asarray(img.shape)/degrad).astype(np.int))
    imin0r = skimage.transform.resize(
        img,
        (np.asarray(img.shape) / degrad).astype(np.int),
        anti_aliasing=False,
        order=1,
        preserve_range=True,
    )
    if sigma is not None:
        from scipy.ndimage.filters import gaussian_filter
        # logger.debug(f"sigma_px={sigma/degrad} ... should be 3")

        imin0r = gaussian_filter(imin0r, sigma / degrad)

    angles = range(-180, 180, 15)
    trax = range(1, imin0r.shape[0], 10)
    tray = range(1, imin0r.shape[1], 10)

    # tr0, tr1, ang0, minval = find_symmetry_parameters(imin0r, trax, tray, angles, debug=debug,
    #                                                   return_criterium_min=True)
    if init_point is None:
        tr0, tr1, ang0, minval = find_symmetry_parameters(
            imin0r, trax, tray, angles, debug=debug, return_criterium_min=True
        )
    else:
        tr0 = int(init_point[0] / degrad)
        tr1 = int(init_point[1] / degrad)
    if init_angle is None:
        ang = ang0
    else:
        ang = int(init_angle) * 2
    # fine measurement
    # the offset is limited to positive values (not really sure why)
    trax = range(np.max([tr0 - 20, 1]), tr0 + 20, 3)
    tray = range(np.max([tr1 - 20, 1]), tr1 + 20, 3)
    angles = list(range(ang - 20, ang + 20, 3))
    if init_angle is None:
        angles += list(range(ang + 180 - 20, ang + 180 + 20, 3))
    # check also the 90 degrees symmetry plane for sure

    tr0, tr1, ang_fine, minval_fine = find_symmetry_parameters(
        imin0r, trax, tray, angles, debug=debug, return_criterium_min=True
    )
    # logger.warning(f"init_angle={init_angle}, angle={ang0}, angle_fine={ang_fine}")

    angle = 90 - ang_fine / 2.0

    # min_angle_diff = np.min(np.abs([
    #     ang_fine - ang0,
    #     ang_fine - ang0 - 180,
    #     ang_fine - ang0 + 180
    #     ]))
    #
    #
    # if min_angle_diff > 25:
    #     logger.warning("Two angle estimation methods have different results.")

    return tr0 * degrad, tr1 * degrad, angle


# Rozděl obraz na půl
def split_with_line(
    point,
    orientation,
    imshape,
    degrees=True,
    voxelsize_0=1.0,
    point_in_positive_halfplane=None,
    return_sgn=False,
):
    """
    :arg point:
    :arg orientation: angle or oriented vector
    :arg degrees: if is set to True inptu angle is expected to be in radians, default True
    """

    if np.isscalar(orientation):
        angle = orientation
        if degrees:
            angle = np.radians(angle)

        # kvadranty
        angle = angle % (2 * np.pi)
        # print np.degrees(angle)
        if (angle > (0.5 * np.pi)) and (angle < (1.5 * np.pi)):
            zn = -voxelsize_0
        else:
            zn = voxelsize_0

        vector = [np.tan(angle), 1]
        # vector = [np.tan(angle), 1]
    else:
        vector = orientation

    vector = vector / np.linalg.norm(vector)
    x, y = np.mgrid[: imshape[0], : imshape[1]]
    #     k = -vector[1]/vector[0]
    #     z = ((k * (x - point[0])) + point[1] - y)
    a = vector[1]
    b = -vector[0]

    c = -a * point[0] - b * point[1]

    if point_in_positive_halfplane is not None:
        xx, yy = point_in_positive_halfplane
        zz = (a * xx + b * yy + c) / (a ** 2 + b ** 2) ** 0.5
        zn = np.sign(zz)

    z = zn * (a * x + b * y + c) / (a ** 2 + b ** 2) ** 0.5
    if return_sgn:
        return z, zn
    else:
        return z


def fill_nan_with_nearest(flat):
    indices = scipy.ndimage.morphology.distance_transform_edt(
        np.isnan(flat), return_indices=True, return_distances=False
    )
    # indices = scipy.ndimage.morphology.distance_transform_edt(flat==0, return_indices=True, return_distances=False)
    ou = flat[(indices[0], indices[1])]
    return ou


def main():

    # logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
        description="Segmentation of bones, lungs and heart."
    )
    parser.add_argument("-i", "--datadir", default=None, help="path to data dir")
    parser.add_argument("-o", "--output", default=None, help="output file")

    parser.add_argument("-d", "--debug", action="store_true", help="run in debug mode")
    parser.add_argument(
        "-ss", "--segspine", action="store_true", help="run spine segmentaiton"
    )
    parser.add_argument(
        "-sb", "--segbody", action="store_true", help="run body segmentaiton"
    )
    parser.add_argument(
        "-exd", "--exampledata", action="store_true", help="run unittest"
    )
    parser.add_argument(
        "-so", "--show_output", action="store_true", help="Show output data in viewer"
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.exampledata:

        args.dcmdir = "../sample_data/liver-orig001.raw"

    #    if dcmdir == None:

    # else:
    # dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
    # data3d, metadata = dcmr.dcm_read_from_dir(args.dcmdir)
    import io3d

    data3d, metadata = io3d.datareader.read(args.datadir, dataplus_format=False)

    bn = BodyNavigation(data3d=data3d, voxelsize_mm=metadata["voxelsize_mm"])

    seg = np.zeros(data3d.shape, dtype=np.int8)
    # sseg.orientation()
    if args.segspine:
        seg += bn.get_spine()
    if args.segspine:
        seg += bn.get_body()

    # print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

    # igc = pycut.ImageGraphCut(data3d, zoom = 0.5)
    # igc.interactivity()

    # igc.make_gc()
    # igc.show_segmentation()

    # volume
    # volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)

    # pyed = sed3.sed3(oseg.data3d, contour = oseg.segmentation)
    # pyed.show()

    if args.show_output:
        import sed3

        sed3.show_slices(data3d, contour=seg)

    # savestring = raw_input ('Save output data? (y/n): ')
    # sn = int(snstring)
    if args.output is not None:  # savestring in ['Y','y']:
        md = {"voxelsize_mm": metadata["voxelsize_mm"], "segmentation": seg}

        io3d.write(data3d, args.output, metadata=md)

    # output = segmentation.vesselSegmentation(oseg.data3d, oseg.orig_segmentation)


def zcopy(slice, shape, dtype=None):
    if dtype is None:
        dtype = slice.dtype
    im3d = np.empty(shape, dtype=dtype)
    for i in range(shape[0]):
        im3d[i, :, :] = slice
    return im3d


if __name__ == "__main__":
    main()
