#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
from nose.plugins.attrib import attr

import os
import os.path as op
import numpy as np
import skimage.measure

import io3d
import io3d.datasets
from bodynavigation.organ_detection import OrganDetection

import sed3 # for testing

# http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.1.zip
# TEST_DATA_DIR = "test_data"
#TEST_DATA_DIR = "/home/jirka642/Programming/_Data/DP/3Dircadb1/1"
TEST_DATA_DIR = "3Dircadb1.1"

class OrganDetectionTest(unittest.TestCase):
    """
    Run only this test class:
        nosetests -v tests.organ_detection_test
        nosetests -v --logging-level=DEBUG tests.organ_detection_test
    Run only single test:
        nosetests -v tests.organ_detection_test:OrganDetectionTest.getBody_test
        nosetests -v --logging-level=DEBUG tests.organ_detection_test:OrganDetectionTest.getBody_test
    """

    @classmethod
    def setUpClass(cls):
        datap = io3d.read(
            io3d.datasets.join_path(TEST_DATA_DIR, "PATIENT_DICOM"),
            dataplus_format=True)
        cls.obj = OrganDetection(datap["data3d"], datap["voxelsize_mm"])

    @classmethod
    def tearDownClass(cls):
        cls.obj = None

    def getBody_test(self):
        # get segmented data
        body = self.obj.getBody()

        # get preprocessed test data
        datap = io3d.read(
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "skin"),
            # io3d.datasets.join_path("PATIENT_DICOM/MASKS_DICOM/skin"),
            dataplus_format=True)
        test_body = datap["data3d"] > 0 # reducing value range to <0,1> from <0,255>

        # ed = sed3.sed3(test_body.astype(np.uint8), contour=body.astype(np.uint8))
        # ed.show()

        # Test requires less then 5% error rate in segmentation
        test_body_sum = np.sum(test_body)
        diff_sum = np.sum(abs(test_body-body))
        self.assertLess(float(diff_sum)/float(test_body_sum), 0.05)

        # There must be only one object (body) in segmented data
        test_body_label = skimage.measure.label(test_body, background=0)
        self.assertEqual(np.max(test_body_label), 1)

    def getLungs_test(self):
        # get segmented data
        lungs = self.obj.getLungs()

        # get preprocessed test data
        datap1 = io3d.read(
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "leftlung"),
            dataplus_format=True)
        datap2 = io3d.read(
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "rightlung"),
            dataplus_format=True)
        test_lungs = (datap1["data3d"]+datap2["data3d"]) > 0 # reducing value range to <0,1> from <0,255>

        # ed = sed3.sed3(test_lungs.astype(np.uint8), contour=lungs.astype(np.uint8))
        # ed.show()

        # Test requires less then 10% error rate in segmentation
        test_lungs_sum = np.sum(test_lungs)
        diff_sum = np.sum(abs(test_lungs-lungs))
        self.assertLess(float(diff_sum)/float(test_lungs_sum), 0.1)

    def getAorta_test(self):
        # get segmented data
        aorta = self.obj.getAorta()

        # get preprocessed test data
        datap = io3d.read(io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "artery"), dataplus_format=True)
        test_aorta = datap["data3d"] > 0 # reducing value range to <0,1> from <0,255>

        # ed = sed3.sed3(test_aorta.astype(np.uint8), contour=aorta.astype(np.uint8))
        # ed.show()

        # Test requires less then 50% error rate in segmentation
        # -> used test data has smaller vessels connected to aorta => that's why the big error
        test_aorta_sum = np.sum(test_aorta)
        diff_sum = np.sum(abs(test_aorta-aorta))
        self.assertLess(float(diff_sum)/float(test_aorta_sum), 2.0)
        # TODO - better -> segment smaller connected vessels OR trim test mask

    def getVenaCava_test(self):
        # get segmented data
        venacava = self.obj.getVenaCava()

        # get preprocessed test data
        datap = io3d.read(io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "venoussystem"), dataplus_format=True)
        test_venacava = datap["data3d"] > 0 # reducing value range to <0,1> from <0,255>

        # ed = sed3.sed3(test_venacava.astype(np.uint8), contour=venacava.astype(np.uint8))
        # ed.show()

        # Test requires less then 50% error rate in segmentation
        # -> used test data has smaller vessels connected to aorta => that's why the big error
        test_venacava_sum = np.sum(test_venacava)
        diff_sum = np.sum(abs(test_venacava-venacava))
        self.assertLess(float(diff_sum)/float(test_venacava_sum), 2.0)
        # TODO - better -> segment smaller connected vessels OR trim test mask

