#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
import sys

import numpy as np
from nose.plugins.attrib import attr

import io3d
import bodynavigation
# from lisa import organ_segmentation
# import pysegbase.dcmreaddata as dcmr
# import lisa.data

TEST_DATA_DIR = "3Dircadb1.1"

# nosetests tests/organ_segmentation_test.py:OrganSegmentationTest.test_create_iparams # noqa

class BodyNavigationTest(unittest.TestCase):
    interactiveTest = False
    verbose = False

    @classmethod
    def setUpClass(cls):
        datap = io3d.read(
            io3d.datasets.join_path(TEST_DATA_DIR, "PATIENT_DICOM"),
            dataplus_format=True)
        cls.obj = bodynavigation.BodyNavigation(datap["data3d"], datap["voxelsize_mm"])
        cls.shape = datap["data3d"].shape

    @classmethod
    def tearDownClass(cls):
        cls.obj = None

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
        self.assertEqual(np.max(seg_spine[30:40, 100:150, 50:150]), 1)
        self.assertEqual(seg_spine[64, 10, 10], 0)
        spine_center = self.obj.spine_center[1:]
        spine_center_expected = [124, 101]
        err = np.linalg.norm(spine_center - spine_center_expected)
        self.assertLessEqual(err, 100)

        spine_dst = self.obj.dist_to_spine()
        self.assertGreater(spine_dst[60,10,10], spine_dst[60, 124, 101])

    def test_get_dists(self):
        dst_lungs = self.obj.dist_to_lungs()
    # @unittest.skipIf(not interactiveTest, "interactive test")

    def test_diaphragm(self):
        dst_diaphragm = self.obj.dist_diaphragm()
        self.assertGreater(dst_diaphragm[10, 120, 120], 10)
        self.assertLess(dst_diaphragm[120, 120, 120], -10)

    def test_dist_sagital(self):
        dst_sagittal = self.obj.dist_sagittal()
        self.assertGreater(dst_sagittal[60, 10, 10], 10)
        self.assertLess(dst_sagittal[60, 10, 500], -10)

    def test_dist_coronal(self):
        dst_coronal = self.obj.dist_coronal()
        # import sed3
        # ed = sed3.sed3(dst_coronal)
        # ed.show()
        self.assertGreater(dst_coronal[60, 10, 10], 50)
        self.assertLess(dst_coronal[60, 500, 10], -50)


if __name__ == "__main__":
    unittest.main()
