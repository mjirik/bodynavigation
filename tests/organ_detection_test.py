#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
from __future__ import print_function   # print("text")
from __future__ import division         # 2/3 == 0.666; 2//3 == 0
from __future__ import absolute_import  # 'import submodule2' turns into 'from . import submodule2'
from builtins import range              # replaces range with xrange

import logging
logger = logging.getLogger(__name__)

import unittest
from nose.plugins.attrib import attr

import io3d
import io3d.datasets
from bodynavigation.organ_detection import OrganDetection
from bodynavigation.tools import readCompoundMask
import bodynavigation.metrics as metrics

import sed3 # for testing

# http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.1.zip
TEST_DATA_DIR = "3Dircadb1.1"

class OrganDetectionTest(unittest.TestCase):
    """
    Run only this test class:
        nosetests -v -s tests.organ_detection_test
        nosetests -v -s --logging-level=DEBUG tests.organ_detection_test
    Run only single test:
        nosetests -v -s tests.organ_detection_test:OrganDetectionTest.getBody_test
        nosetests -v -s --logging-level=DEBUG tests.organ_detection_test:OrganDetectionTest.getBody_test
    """

    # Minimal dice coefficients
    GET_BODY_DICE = 0.95
    GET_LUNGS_DICE = 0.95
    GET_BONES_DICE =  0.75 # test data don't have segmented whole bones, missing center volumes
    GET_KIDNEYS_DICE = 0.75
    GET_AORTA_DICE = 0.25 # TODO - used test data has smaller vessels connected to aorta => that's why the big error margin
    GET_VENACAVA_DICE = 0.25 # TODO - used test data has smaller vessels connected to aorta => that's why the big error margin


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
        mask = self.obj.getBody()

        # get preprocessed test data
        test_mask, _ = readCompoundMask([
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "skin"),
            ])

        # Test requires at least ??% of correct segmentation
        dice = metrics.dice(test_mask, mask)
        print("getBody(), Dice coeff: %s" % str(dice))
        self.assertGreater(dice, self.GET_BODY_DICE)

    def getLungs_test(self):
        # get segmented data
        mask = self.obj.getLungs()

        # get preprocessed test data
        test_mask, _ = readCompoundMask([
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "leftlung"),
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "rightlung"),
            ])

        # Test requires at least ??% of correct segmentation
        dice = metrics.dice(test_mask, mask)
        print("getLungs(), Dice coeff: %s" % str(dice))
        self.assertGreater(dice, self.GET_LUNGS_DICE)

    def getBones_test(self):
        # get segmented data
        mask = self.obj.getBones()

        # get preprocessed test data
        test_mask, _ = readCompoundMask([
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "bone"),
            ])

        # Test requires at least ??% of correct segmentation
        dice = metrics.dice(test_mask, mask)
        print("getBones(), Dice coeff: %s" % str(dice))
        self.assertGreater(dice, self.GET_BONES_DICE)

    def getAorta_test(self):
        # get segmented data
        mask = self.obj.getAorta()

        # get preprocessed test data
        test_mask, _ = readCompoundMask([
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "artery"),
            ])

        # Test requires at least ??% of correct segmentation
        dice = metrics.dice(test_mask, mask)
        print("getAorta(), Dice coeff: %s" % str(dice))
        self.assertGreater(dice, self.GET_AORTA_DICE)
        # TODO - better -> segment smaller connected vessels OR trim test mask

    def getVenaCava_test(self):
        # get segmented data
        mask = self.obj.getVenaCava()

        # get preprocessed test data
        test_mask, _ = readCompoundMask([
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "venoussystem"),
            ])

        # Test requires at least ??% of correct segmentation
        dice = metrics.dice(test_mask, mask)
        print("getVenaCava(), Dice coeff: %s" % str(dice))
        self.assertGreater(dice, self.GET_VENACAVA_DICE)
        # TODO - better -> segment smaller connected vessels OR trim test mask

    def getKidneys_test(self):
        # get segmented data
        mask = self.obj.getKidneys()

        # get preprocessed test data
        test_mask, _ = readCompoundMask([
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "leftkidney"),
            io3d.datasets.join_path(TEST_DATA_DIR, "MASKS_DICOM", "rightkidney"),
            ])

        # Test requires at least ??% of correct segmentation
        dice = metrics.dice(test_mask, mask)
        print("getKidneys(), Dice coeff: %s" % str(dice))
        self.assertGreater(dice, self.GET_KIDNEYS_DICE)


