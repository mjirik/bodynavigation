#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
from nose.plugins.attrib import attr

import os
import numpy as np
import skimage.measure

import io3d
from bodynavigation import BodyNavigation

# http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.1.zip
TEST_DATA_DIR = "test_data"

class SegmentationTest(unittest.TestCase):

    def bodySegmentation_test(self):
        # get segmented data
        datap = io3d.read(os.path.join(TEST_DATA_DIR, "PATIENT_DICOM"), dataplus_format=True)
        bn = BodyNavigation(datap["data3d"], datap["voxelsize_mm"])
        bn_body = bn.get_body()

        # get preprocessed test data
        datap = io3d.read(os.path.join(TEST_DATA_DIR, "MASKS_DICOM", "skin"), dataplus_format=True)
        test_body = datap["data3d"] > 0 # reducing value range to <0,1> from <0,255>

        # Test requires less then 5% error rate in segmentation
        test_body_sum = np.sum(test_body)
        diff_sum = np.sum(abs(test_body-bn_body))
        self.assertLess(float(diff_sum)/float(test_body_sum), 0.05)

        # There must be only one object (body) in segmented data
        test_body_label = skimage.measure.label(test_body, background=0)
        self.assertEqual(np.max(test_body_label), 1)

    def lungsSegmentation_test(self):
        # get segmented data
        datap = io3d.read(os.path.join(TEST_DATA_DIR, "PATIENT_DICOM"), dataplus_format=True)
        bn = BodyNavigation(datap["data3d"], datap["voxelsize_mm"])
        bn_lungs = bn.get_lungs()

        # get preprocessed test data
        datap1 = io3d.read(os.path.join(TEST_DATA_DIR, "MASKS_DICOM", "leftlung"), dataplus_format=True)
        datap2 = io3d.read(os.path.join(TEST_DATA_DIR, "MASKS_DICOM", "rightlung"), dataplus_format=True)
        test_lungs = (datap1["data3d"]+datap2["data3d"]) > 0 # reducing value range to <0,1> from <0,255>

        # Test requires less then 5% error rate in segmentation
        test_lungs_sum = np.sum(test_lungs)
        diff_sum = np.sum(abs(test_lungs-bn_lungs))
        self.assertLess(float(diff_sum)/float(test_lungs_sum), 0.05)

    @unittest.skip("BodyNavigation.get_aorta() Not Implemented")
    def aortaSegmentation_test(self):
        # get segmented data
        datap = io3d.read(os.path.join(TEST_DATA_DIR, "PATIENT_DICOM"), dataplus_format=True)
        bn = BodyNavigation(datap["data3d"], datap["voxelsize_mm"])
        bn_aorta = bn.get_aorta()

        # get preprocessed test data
        datap = io3d.read(os.path.join(TEST_DATA_DIR, "MASKS_DICOM", "artery"), dataplus_format=True)
        test_aorta = datap["data3d"] > 0 # reducing value range to <0,1> from <0,255>

        # Test requires less then 5% error rate in segmentation
        test_aorta_sum = np.sum(test_aorta)
        diff_sum = np.sum(abs(test_aorta-bn_aorta))
        self.assertLess(float(diff_sum)/float(test_aorta_sum), 0.05)

    @unittest.skip("BodyNavigation.get_vena_cava() Not Implemented")
    def venaCavaSegmentation_test(self):
        # get segmented data
        datap = io3d.read(os.path.join(TEST_DATA_DIR, "PATIENT_DICOM"), dataplus_format=True)
        bn = BodyNavigation(datap["data3d"], datap["voxelsize_mm"])
        bn_venacava = bn.get_vena_cava()

        # get preprocessed test data
        datap = io3d.read(os.path.join(TEST_DATA_DIR, "MASKS_DICOM", "venoussystem"), dataplus_format=True)
        test_venacava = datap["data3d"] > 0 # reducing value range to <0,1> from <0,255>

        # Test requires less then 5% error rate in segmentation
        test_venacava_sum = np.sum(test_venacava)
        diff_sum = np.sum(abs(test_venacava-bn_venacava))
        self.assertLess(float(diff_sum)/float(test_venacava_sum), 0.05)

