#! /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
# import logging
#
# logger = logging.getLogger(__name__)

import unittest
import sys

import numpy as np
import pytest

import bodynavigation

# from lisa import organ_segmentation
# import pysegbase.dcmreaddata as dcmr
# import lisa.data

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import glob

import io3d
import io3d.datasets
# import sed3
# import SimpleITK as sitk
# sys.path.append(Path("~/projects/bodynavigation").expanduser())
import bodynavigation

import matplotlib.pyplot as plt

# TEST_DATA_DIR = "3Dircadb1.1"
dataset = "3Dircadb1"
spine_center_working = [60, 124, 101]
ircad1_spine_center_idx = [120, 350, 260]
ircad1_liver_center_idx = [25, 220, 180]
# nosetests tests/organ_segmentation_test.py:OrganSegmentationTest.test_create_iparams # noqa


class BodyNavigationTest(unittest.TestCase):
    interactiveTest = False
    verbose = False

    @classmethod
    def setUpClass(self):
        # datap = io3d.read(
        #     io3d.datasets.join_path(TEST_DATA_DIR, "PATIENT_DICOM"),
        #     dataplus_format=True,
        # )
        # pth = io3d.datasets.get_dataset_path(dataset, 'data3d', 1)
        # print(f"pth={pth}")
        datap = io3d.read_dataset(dataset, 'data3d', 1, orientation_axcodes='SPL')

        self.obj:bodynavigation.BodyNavigation = bodynavigation.BodyNavigation(datap["data3d"], datap["voxelsize_mm"])
        self.data3d = datap["data3d"]
        self.shape = datap["data3d"].shape

    @classmethod
    def tearDownClass(self):
        self.obj = None

    def test_get_body(self):
        seg_body = self.obj.get_body()
        self.assertEqual(seg_body[64, 256, 256], 1)
        self.assertEqual(seg_body[64, 10, 10], 0)
        self.assertGreaterEqual(self.shape[0], seg_body.shape[0])

        # check whether inside is positive and outside is zero
        dst_surface = self.obj.dist_to_surface()
        # import sed3
        # ed = sed3.sed3(dst_surface)
        # ed.show()
        self.assertGreater(dst_surface[50, 124, 121], 5)
        self.assertEqual(dst_surface[50, 10, 10], 0)

    def test_get_spine(self):
        seg_spine = self.obj.get_spine()

        # import sed3
        # ed = sed3.sed3(seg_spine)
        # ed.show()
        self.assertEqual(np.max(seg_spine[30:40, 300:400, 200:300]), 1)
        self.assertEqual(seg_spine[64, 10, 10], 0)
        spine_center = self.obj.get_center()[1:]
        spine_center_expected = [27, 47]
        # spine_center_working_expected = [124, 101]
        err = np.linalg.norm(spine_center - spine_center_expected)
        self.assertLessEqual(err, 100)

        spine_dst = self.obj.dist_to_spine()
        self.assertGreater(spine_dst[60, 10, 10], spine_dst[60, 124, 101])

    def test_get_diapghragm_axial(self):
        max_error_mm = 5
        i, mask = self.obj.get_diaphragm_axial_position_index(return_in_working_voxelsize=False, return_mask=True)
        assert self.obj.data3dr.shape[0] == mask.shape[0], "Shape of mask should be always in resized size"

        dst = self.obj.dist_to_diaphragm_axial()
        randi = np.random.randint(1, mask.shape[0])
        assert dst[randi, 0, 0] == dst[randi, -1, -1], "Distances in one slide should be equal"
        assert dst[0, 0, 0] != dst[randi, -1, -1], "Distances in neighboring slides should be different"
        datap2 = io3d.datasets.read_dataset(dataset, "liver", 1, orientation_axcodes='SPL')
        ii_liver = np.min(np.nonzero(datap2.data3d)[0])
        max_error_px = max_error_mm / datap2.voxelsize_mm[0]

        assert pytest.approx(ii_liver, max_error_px) == i, "Index of end of the liver should be close to the detected diaphragm level index"

        assert dst.shape[0] == self.obj.orig_shape[0], "shape is in orig shape"

        assert pytest.approx(dst[int(i),0,0], max_error_mm) == 0, "Distance in detected liver slide should be zero"
        assert pytest.approx(dst[ii_liver,0,0], max_error_mm) == 0, "Distance in annotated liver slide should be close to zero"


        dst_wvs = self.obj.dist_to_diaphragm_axial(return_in_working_voxelsize=True)
        assert dst_wvs.shape[0] == self.obj.data3dr.shape[0]


    def test_get_dists(self):
        dst_lungs = self.obj.dist_to_lungs()

    # @unittest.skipIf(not interactiveTest, "interactive test")

    def test_diaphragm_profile_image(self):
        profile, gradient = self.obj.get_diaphragm_profile_image_with_empty_areas(
            return_gradient_image=True
        )
        self.assertGreater(
            np.max(gradient),
            self.obj.GRADIENT_THRESHOLD,
            "Gradient threshold is too low",
        )
        self.assertLess(
            np.min(gradient),
            -self.obj.GRADIENT_THRESHOLD,
            "Gradient threshold is to low",
        )

        self.assertGreater(
            np.nanmax(profile) - np.nanmin(profile),
            5,
            "Low and high diaphragm level should be at least 5 slices",
        )
        import matplotlib.pyplot as plt

        # import sed3
        # ed = sed3.sed3(gradient)
        # ed.show()
        # plt.imshow(profile)
        # plt.colorbar()
        # plt.show()
        # grt = gradient > self.obj.GRADIENT_THRESHOLD

    def test_diaphragm(self):
        # import sed3
        # ed = sed3.sed3(self.data3d, contour=self.obj.get_lungs())
        # ed.show()
        import matplotlib.pyplot as plt

        profile_with_nan, gradient = self.obj.get_diaphragm_profile_image_with_empty_areas(
            return_gradient_image=True
        )
        self.assertGreater(
            np.sum(np.isnan(profile_with_nan)),
            np.prod(profile_with_nan.shape) * 0.10,
            "Expected at least 10% NaN's in profile image",
        )

        self.assertGreater(
            np.nanmean(profile_with_nan),
            5,
            "The diaphragm is expected to be more than 5 slices in the data",
        )
        self.assertLess(
            np.nanmean(profile_with_nan),
            self.obj.lungs.shape[0] - 5,
            "The diaphragm is expected to be more than 5 slices in the data",
        )
        # plt.imshow(profile_with_nan)
        # plt.colorbar()
        # plt.show()
        profile, preprocessed_profile = self.obj.get_diaphragm_profile_image(
            return_preprocessed_image=True
        )
        self.assertEqual(
            np.sum(np.isnan(profile)),
            0,
            "Expected 0 NaN's in postprocessed profile image",
        )
        self.assertGreater(
            np.nanmean(profile),
            5,
            "The diaphragm is expected to be more than 5 slices in the data",
        )
        self.assertLess(
            np.nanmean(profile),
            self.obj.lungs.shape[0] - 5,
            "The diaphragm is expected to be more than 5 slices in the data",
        )
        # plt.imshow(preprocessed_profile)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(profile)
        # plt.colorbar()
        # plt.show()
        # print(dst_diaphragm.shape, self.data3d.shape)
        # import sed3
        # ed = sed3.sed3(dst_diaphragm)
        # ed.show()
        # ed = sed3.sed3(self.obj.get_diaphragm_mask())
        # ed.show()
        # ed = sed3.sed3(self.data3d, contour=self.obj.get_diaphragm_mask())
        # ed.show()
        dst_diaphragm = self.obj.dist_to_diaphragm()
        # above diaphragm it should be positive
        # plt.imshow(self.data3d[0,:,:])
        # plt.show()
        # plt.imshow(dst_diaphragm[0,:,:])
        # plt.colorbar()
        # plt.show()
        # plt.imshow(dst_diaphragm[:,250,:])
        # plt.colorbar()
        # plt.show()
        self.assertGreater(
            dst_diaphragm[0, 500, 10],
            2,
            "Diaphragm should be at least few mm from the top of the image",
        )
        # unter diaphragm
        self.assertLess(dst_diaphragm[120, 250, 250], -20)

    def test_dist_sagital(self):
        dst_sagittal = self.obj.dist_to_sagittal()
        axis=0
        import sed3
        dst = dst_sagittal
        # sed3.show_slices(data3d=self.data3d, contour=dst>0, slice_number=6, axis=axis)
        sed3.show_slices(data3d=dst, contour=self.data3d>0, slice_number=6, axis=axis)
        self.assertGreater(dst_sagittal[60, 10, 10], 10)
        self.assertLess(dst_sagittal[60, 10, 500], -10)

    def test_dist_coronal(self):
        dst_coronal = self.obj.dist_to_coronal()
        # import sed3
        # ed = sed3.sed3(dst_coronal)
        # ed.show()
        self.assertGreater(dst_coronal[60, 10, 10], 50)
        self.assertLess(dst_coronal[60, 500, 10], -50)

    def test_dist_axial(self):
        dst = self.obj.dist_to_axial()
        import sed3
        # ed = sed3.sed3(dst)
        # ed.show()
        self.assertLess(dst[0, 250, 250], 10)
        self.assertGreater(dst[100, 250, 250], 30)

    # @unittest.skip("problem with brodcast together different shapes")
    def test_chest(self):
        dst = self.obj.dist_to_chest()
        # self.data3d[
        #     # :,
        #     ircad1_liver_center_idx[0]:ircad1_liver_center_idx[0] + 20,
        #     ircad1_liver_center_idx[1]:ircad1_liver_center_idx[1] + 20,
        #     ircad1_liver_center_idx[2]:ircad1_liver_center_idx[2] + 20,
        # ] = 1000
        # import sed3
        # ed = sed3.sed3(self.data3d)
        # ed = sed3.sed3(self.data3d, contour=(dst > 0))
        # ed = sed3.sed3(self.dst, contour=(dst > 0))
        # ed.show()

        self.assertLess(dst[10, 10, 10], -10)
        self.assertGreater(
            dst[
                ircad1_liver_center_idx[0],
                ircad1_liver_center_idx[1],
                ircad1_liver_center_idx[2],
            ],
            10,
        )

    # @unittest.skip("problem with brodcast together different shapes")
    def test_ribs(self):
        dst = self.obj.dist_to_ribs()
        #
        # import sed3
        # # ed = sed3.sed3(self.data3d)
        # ed = sed3.sed3(self.data3d, contour=(dst > 0))
        # # ed = sed3.sed3(self.dst, contour=(dst > 0))
        # ed.show()

        self.assertGreater(dst[10, 10, 10], 10)
        self.assertGreater(
            dst[
                ircad1_liver_center_idx[0],
                ircad1_liver_center_idx[1],
                ircad1_liver_center_idx[2],
            ],
            10,
        )
        # import sed3

    def test_diaphragm_martin(self):
        # bn = bodynavigation.BodyNavigation(use_new_get_lungs_setup=True)
        self.obj.use_new_get_lungs_setup = True
        binary_lungs = self.obj.get_lungs_martin()
        dst_diaphragm = self.obj.dist_to_diaphragm()
        # import sed3
        # ed = sed3.sed3(dst_diaphragm)
        # ed.show()
        # above diaphragm
        self.assertGreater(dst_diaphragm[0, 500, 10], 0)
        # unter diaphragm
        self.assertLess(dst_diaphragm[120, 250, 250], -20)
        self.obj.use_new_get_lungs_setup = False



def test_sagital():

    datap = io3d.datasets.read_dataset("3Dircadb1", 'data3d', 1)
    data3d = datap["data3d"][:110]
    voxelsize_mm = datap["voxelsize_mm"]

    def show_dists(dist, i=100, j=200):
        fig, axs = plt.subplots(
            2, 2,
            #         sharey=True,
            figsize=[15, 12])
        axs = axs.flatten()
        axs[0].imshow(data3d[i, :, :], cmap='gray')

        axs[1].imshow(dist[i, :, :])
        axs[1].contour(dist[i, :, :] > 0)
        axs[2].imshow(data3d[:, j, :], cmap='gray')
        axs[3].imshow(dist[:, j, :])
        axs[3].contour(dist[:, j, :] > 0)

        for ax in axs:
            ax.axis('off')

    ss = bodynavigation.body_navigation.BodyNavigation(data3d, voxelsize_mm)
    # dist = ss.dist_sagittal()
    dist = ss.dist_coronal()
    # dist = ss.dist_to_surface()
    # show_dists(dist)
    # plt.show()
    assert dist[0, 255, 255] > 0
    assert dist[0, 400, 400] < 0

def test_spine_all_spines_in_dataset():
    one_i = None
    for i in range(1, 2):
    # for i in range(1, 21):
    # one_i = 8
    # for i in range(one_i, one_i + 1):
        datap = io3d.datasets.read_dataset("3Dircadb1", 'data3d', i)
        # datap = io3d.datasets.read_dataset("sliver07",'data3d', i)
        data3d = datap["data3d"]
        # data3d = datap["data3d"][:110]
        voxelsize_mm = datap["voxelsize_mm"]

        def show_dists(dist, i=63, j=200):
            fig, axs = plt.subplots(
                2, 2,
                #         sharey=True,
                figsize=[15, 12])
            axs = axs.flatten()
            axs[0].imshow(data3d[i, :, :], cmap='gray')

            axs[1].imshow(dist[i, :, :])
            axs[1].contour(dist[i, :, :] > 0)
            axs[2].imshow(data3d[:, j, :], cmap='gray')
            axs[3].imshow(dist[:, j, :])
            axs[3].contour(dist[:, j, :] > 0)

            for ax in axs:
                ax.axis('off')

        ss = bodynavigation.body_navigation.BodyNavigation(data3d, voxelsize_mm)
        if one_i:
            ss.debug = True
        # dist = ss.dist_sagittal()
        dist = ss.dist_to_spine()
        # dist = ss.dist_to_surface()
        # print(ss.spine.dtype)
        # show_dists(dist)
        # plt.figure()
        # plt.imshow(np.max(ss.spine, axis=0))
        # plt.show()

        # check standard deviation of spine pixels coordinates in voxelsize_mm. It should be less than 10 px along x,y
        spine_coords_std = np.std(np.nonzero(ss.spine), 1)
        assert spine_coords_std[0] > 5, "Pixels should be wide spread alogn Z axis"
        assert spine_coords_std[1] < 15
        assert spine_coords_std[2] < 15
        # assert dist[0, 255, 255] > 0
        # assert dist[0, 400, 400] < 0

# def test_bone():
#
#     for i in range(1, 1):
#         datap = io3d.datasets.read_dataset("3Dircadb1", 'data3d', i)
#         # datap = io3d.datasets.read_dataset("sliver07",'data3d', i)
#         # data3d = datap["data3d"][:110]
#         data3d = datap["data3d"][:110]
#         voxelsize_mm = datap["voxelsize_mm"]
#
#         def show_dists(dist, i=100, j=200):
#             fig, axs = plt.subplots(
#                 2, 2,
#                 #         sharey=True,
#                 figsize=[15, 12])
#             axs = axs.flatten()
#             axs[0].imshow(data3d[i, :, :], cmap='gray')
#
#             axs[1].imshow(dist[i, :, :])
#             axs[1].contour(dist[i, :, :] > 0)
#             axs[2].imshow(data3d[:, j, :], cmap='gray')
#             axs[3].imshow(dist[:, j, :])
#             axs[3].contour(dist[:, j, :] > 0)
#
#             for ax in axs:
#                 ax.axis('off')
#
#         ss = bodynavigation.body_navigation.BodyNavigation(data3d, voxelsize_mm)
#         # dist = ss.dist_sagittal()
#         dist = ss.dist_to_spine()
#         # dist = ss.dist_to_surface()
#         # show_dists(dist)
#         # plt.show()
#         # check standard deviation of spine pixels coordinates in voxelsize_mm. It should be less than 10 px along x,y
#         spine_coords_std = np.std(np.nonzero(ss.spine), 1)
#         assert spine_coords_std[0] > 10, "Pixels should be wide spread alogn z axis"
#         assert spine_coords_std[1] < 10
#         assert spine_coords_std[2] < 10
#         # assert dist[0, 255, 255] > 0
#         # assert dist[0, 400, 400] < 0

def test_sagit_by_spine_on_both_sides_of_sagitt_in_whole_dataset():
    r_spine_mm = 10
    one_i = None
    debug = False
    for i in range(1, 2):
    # for i in range(1, 21):
    # one_i = 1
    # for i in range(one_i, one_i + 1):
        datap = io3d.datasets.read_dataset("3Dircadb1", 'data3d', i)
        # datap = io3d.datasets.read_dataset("sliver07",'data3d', i)
        data3d = datap["data3d"]
        # data3d = datap["data3d"][:110]
        voxelsize_mm = datap["voxelsize_mm"]

        def show_dists(dist, i=63, j=200, contour=None):
            if contour is None:
                contour = dist > 0
            fig, axs = plt.subplots(
                2, 2,
                #         sharey=True,
                figsize=[15, 12])
            axs = axs.flatten()
            axs[0].imshow(data3d[i, :, :], cmap='gray')

            axs[1].imshow(dist[i, :, :])
            axs[1].contour(contour[i, :, :])
            axs[2].imshow(data3d[:, j, :], cmap='gray')
            axs[3].imshow(dist[:, j, :])
            axs[3].contour(contour[:, j, :])

            for ax in axs:
                ax.axis('off')

        ss = bodynavigation.body_navigation.BodyNavigation(data3d, voxelsize_mm)
        ss.debug = debug
        distsp = ss.dist_to_spine()
        dist = ss.dist_to_sagittal()
        # dist = ss.dist_to_surface()
        if debug:
            print(ss.spine.dtype)
            r_spine_mm = 10
            show_dists(distsp - 10,
                       contour=(dist > 0).astype(np.uint8) + (distsp > r_spine_mm).astype(np.uint8)
                       )
            plt.suptitle(f"i={i}")
            plt.show()

        # check if spine is on both sides of sagittal plane
        sagittal_dists_in_spine = dist[distsp < r_spine_mm]
        assert np.max(sagittal_dists_in_spine) > 0
        assert np.min(sagittal_dists_in_spine) < 0

def test_sagittal_on_data_augmented_by_rotation():
    import scipy
    import itertools
    angle = 130
    # angle = 0
    debug = False
    i = 10
    for dataset, i in itertools.product([
        "3Dircadb1",
        # "sliver07"
    ], list(range(1,2))):
        logger.debug(f"{dataset} {i}")
        datap = io3d.datasets.read_dataset(dataset, 'data3d', i, orientation_axcodes="SPL")
        data3d = datap["data3d"][50:,:,:]
        # data3d = datap["data3d"][:,:,:]
        voxelsize_mm = datap["voxelsize_mm"]
        # data3d = datap["data3d"][:110]
        imr = scipy.ndimage.rotate(data3d, angle, axes=[1,2], cval=-1000, reshape=False)
        # import sed3
        # ed = sed3.sed3(imr)
        # ed.show()
        ss = bodynavigation.body_navigation.BodyNavigation(imr, voxelsize_mm)
        ss.debug = debug
        dist = ss.dist_to_sagittal()
        translated_angle = 90 - angle
        plt.imshow(imr[20, :, :] + 10* dist[20,:,:], cmap='gray')
        plt.contour((dist>0)[20, :, :])
        plt.suptitle(f"{dataset} {i} set_angle={angle}, translated_angle={translated_angle}, angle_estimation={ss.angle}")

        plt.show()
        min_diff = np.min(np.abs([
            ss.angle - translated_angle,
            ss.angle - translated_angle + 180,
            ss.angle - translated_angle - 180,

        ]))
        assert min_diff == pytest.approx(0, abs=10)

# class BodyNavigationTestRotated180(unittest.TestCase):
#     interactiveTest = False
#     verbose = False
#
#     @classmethod
#     def setUpClass(self):
#         # datap = io3d.read(
#         #     io3d.datasets.join_path(TEST_DATA_DIR, "PATIENT_DICOM"),
#         #     dataplus_format=True,
#         # )
#         # pth = io3d.datasets.get_dataset_path(dataset, 'data3d', 1)
#         # print(f"pth={pth}")
#         dataset = "sliver07"
#         dataset = "3Dircadb1"
#         datap = io3d.read_dataset(dataset, 'data3d', 1, orientation_axcodes='SPL')
#
#         data3d = datap['data3d']
#         data3d = np.rot90(data3d,k=0, axes=(1,2))
#         self.obj:bodynavigation.BodyNavigation = bodynavigation.BodyNavigation(data3d, datap["voxelsize_mm"])
#         self.data3d = data3d
#         self.shape = data3d.shape
#
#     @classmethod
#     def tearDownClass(self):
#         self.obj = None
#
#
#     def test_dist_sagital(self):
#         dst_sagittal = self.obj.dist_to_sagittal()
#         axis = 0
#         import sed3
#         dst = dst_sagittal
#         sed3.show_slices(data3d=dst, contour=self.data3d > 0, slice_number=6, axis=axis)
#         plt.figure()
#         sed3.show_slices(data3d=self.data3d, contour=dst>0, slice_number=6, axis=axis)
#         self.assertLess(dst_sagittal[60, 10, 10], 10)
#         self.assertGreater(dst_sagittal[60, 10, 500], -10)


if __name__ == "__main__":
    unittest.main()
