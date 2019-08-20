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

import unittest

# from nose.tools import nottest
import pytest

import io3d
import io3d.datasets
from bodynavigation.organ_detection import OrganDetection
from bodynavigation.tools import readCompoundMask
import bodynavigation.metrics as metrics
from bodynavigation.files import loadDatasetsInfo, joinDatasetPaths


# http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.1.zip
ROOT_PATH = io3d.datasets.dataset_path()
DATASET_NAME = "3Dircadb1.1"


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
    DICE = {
        "body": 0.95,
        "lungs": 0.95,
        "bones": 0.75,  # test data don't have segmented whole bones, missing center volumes
        "vessels": 0.50,  # used test data has smaller vessels connected to aorta/venacava => that's why the big error margin
        "kidneys": 0.70,
        "liver": 0.75,
        "spleen": 0.75,
    }

    @classmethod
    def setUpClass(cls):
        # hide io3d logger
        logging.getLogger("io3d").setLevel(logging.WARNING)

        # init dataset information
        cls.dataset = loadDatasetsInfo()[DATASET_NAME]
        cls.dataset = joinDatasetPaths(cls.dataset, ROOT_PATH)

        # init OrganDetection object
        datap = io3d.read(cls.dataset["CT_DATA_PATH"], dataplus_format=True)
        cls.obj = OrganDetection(datap["data3d"], datap["voxelsize_mm"])

    @classmethod
    def tearDownClass(cls):
        cls.obj = None

    @pytest.mark.skip()
    def _genericMaskTest(self, part):
        # get segmented data
        mask = self.obj.getPart(part)
        # get preprocessed test data
        test_mask, _ = readCompoundMask(self.dataset["MASKS"][part])
        # Test requires at least ??% of correct segmentation
        dice = metrics.dice(test_mask, mask)
        print("%s, dice coeff: %s" % (part, str(dice)))
        self.assertGreater(dice, self.DICE[part])

    def getBody_test(self):
        self._genericMaskTest("body")

    def getLungs_test(self):
        self._genericMaskTest("lungs")

    def getBones_test(self):
        self._genericMaskTest("bones")

    def getVessels_test(self):
        self._genericMaskTest("vessels")

    def getKidneys_test(self):
        self._genericMaskTest("kidneys")

    def getLiver_test(self):
        self._genericMaskTest("liver")

    def getSpleen_test(self):
        self._genericMaskTest("spleen")
