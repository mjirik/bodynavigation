#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
# from __future__ import print_function   # print("text")
# from __future__ import division         # 2/3 == 0.666; 2//3 == 0
# from __future__ import absolute_import  # 'import submodule2' turns into 'from . import submodule2'
# from builtins import range              # replaces range with xrange

import logging
logger = logging.getLogger(__name__)
import unittest
import numpy as np
import io3d
import bodynavigation as bn

from bodynavigation.results_drawer import ResultsDrawer
from bodynavigation.organ_detection import OrganDetection

TEST_DATA_DIR = "3Dircadb1.1"
DATA_PATH = "3Dircadb1.1/PATIENT_DICOM"

class VisualizationTest(unittest.TestCase):
    """
    Run only this test class:
        nosetests -v -s tests.organ_detection_test
        nosetests -v -s --logging-level=DEBUG tests.organ_detection_test
    Run only single test:
        nosetests -v -s tests.organ_detection_test:OrganDetectionTest.getBody_test
        nosetests -v -s --logging-level=DEBUG tests.organ_detection_test:OrganDetectionTest.getBody_test
    """


    def basic_drawer_test(self):
        # datap = io3d.read(
        #     io3d.datasets.join_path(TEST_DATA_DIR, "PATIENT_DICOM"),
        #     dataplus_format=True)
        data3d, metadata = io3d.datareader.read(io3d.datasets.join_path(DATA_PATH), dataplus_format=False)
        voxelsize = metadata["voxelsize_mm"]
        # obj = OrganDetection(data3d, voxelsize)
        # masks = [ obj.getPart(p) for p in ["bones","lungs","kidneys"] ]
        object1 = np.zeros_like(data3d)
        object1[40:80, 140:200, 140:200] = 1
        object1[50:90, 180:210, 150:220] = 2
        object2 = np.zeros_like(data3d)
        object2[60:95, 135:200, 70:100] = 1
        masks = [object1, object2]
        # bones_stats = obj.analyzeBones()
        # points = [bones_stats["spine"], bones_stats["hips_start"]]

        points = [[(10,10,10), (50, 150, 150)], [(20, 10, 20), (20, 15, 25), (25, 10, 25)]]
        rd = ResultsDrawer(default_volume_alpha=100)
        img = rd.drawImageAutocolor(data3d, voxelsize, volumes=masks, points=points)
        # img.show()
        self.assertGreater(img.width, 100)
        self.assertGreater(img.height, 100)

    def basic_drawer_complex_test(self):
        # datap = io3d.read(
        #     io3d.datasets.join_path(TEST_DATA_DIR, "PATIENT_DICOM"),
        #     dataplus_format=True)
        data3d, metadata = io3d.datareader.read(io3d.datasets.join_path(DATA_PATH), dataplus_format=False)
        voxelsize = metadata["voxelsize_mm"]
        obj = OrganDetection(data3d, voxelsize)
        masks = [ obj.getPart(p) for p in ["bones","lungs","kidneys"] ]
        bones_stats = obj.analyzeBones()
        points = [bones_stats["spine"], bones_stats["hips_start"]]
        print(points)
        logger.debug(points)

        rd = ResultsDrawer(default_volume_alpha=100)
        img = rd.drawImageAutocolor(data3d, voxelsize, volumes=masks, points=points)
        # img.show()
