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
        data3d, metadata = io3d.datareader.read(
            io3d.datasets.join_path(DATA_PATH), dataplus_format=False
        )
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

        # every points is represented in visualization just by one voxel
        data = np.zeros_like(data3d)
        data[30:40, 350:360, 260:270] = 1
        pts0 = np.nonzero(data)
        points = [list(zip(pts0[0], pts0[1], pts0[2]))]
        # np.nonzero(data)
        # points = [[(10,10,10), (50, 150, 150)], [(20, 10, 20), (20, 15, 25), (25, 10, 25)]]
        points = [
            [
                (0, 362, 263),
                (1, 361, 263),
                (2, 360, 263),
                (3, 359, 263),
                (4, 359, 264),
                (5, 358, 264),
                (6, 357, 264),
                (7, 356, 264),
                (8, 356, 264),
                (9, 355, 265),
                (10, 354, 265),
                (11, 353, 265),
                (12, 353, 265),
                (13, 352, 265),
                (14, 351, 265),
                (15, 350, 265),
                (16, 349, 265),
                (17, 349, 266),
                (18, 348, 266),
                (19, 347, 266),
                (20, 346, 266),
                (21, 345, 266),
                (22, 345, 266),
                (23, 344, 266),
                (24, 343, 266),
                (25, 342, 266),
                (26, 341, 266),
                (27, 341, 266),
                (28, 340, 266),
                (29, 339, 266),
                (30, 338, 266),
                (31, 337, 266),
                (32, 337, 266),
                (33, 336, 266),
                (34, 335, 266),
                (35, 334, 266),
                (36, 334, 266),
                (37, 333, 266),
                (38, 332, 266),
                (39, 331, 266),
                (40, 331, 265),
                (41, 330, 265),
                (42, 329, 265),
                (43, 328, 265),
                (44, 328, 265),
                (45, 327, 265),
                (46, 326, 265),
                (47, 325, 265),
                (48, 325, 265),
                (49, 324, 265),
                (50, 323, 265),
                (51, 323, 264),
                (52, 322, 264),
                (53, 321, 264),
                (54, 321, 264),
                (55, 320, 264),
                (56, 319, 264),
                (57, 319, 264),
                (58, 318, 264),
                (59, 317, 264),
                (60, 317, 263),
                (61, 316, 263),
                (62, 316, 263),
                (63, 315, 263),
                (64, 315, 263),
                (65, 314, 263),
                (66, 313, 263),
                (67, 313, 262),
                (68, 312, 262),
                (69, 312, 262),
                (70, 312, 262),
                (71, 311, 262),
                (72, 311, 262),
                (73, 310, 262),
                (74, 310, 262),
                (75, 309, 261),
                (76, 309, 261),
                (77, 309, 261),
                (78, 308, 261),
                (79, 308, 261),
                (80, 308, 261),
                (81, 307, 261),
                (82, 307, 261),
                (83, 307, 260),
                (84, 307, 260),
                (85, 306, 260),
                (86, 306, 260),
                (87, 306, 260),
                (88, 306, 260),
                (89, 306, 260),
                (90, 305, 260),
                (91, 305, 260),
                (92, 305, 259),
                (93, 305, 259),
                (94, 305, 259),
                (95, 305, 259),
                (96, 305, 259),
                (97, 305, 259),
                (98, 305, 259),
                (99, 305, 259),
                (100, 305, 259),
                (101, 305, 259),
                (102, 306, 259),
                (103, 306, 259),
                (104, 306, 259),
                (105, 306, 259),
                (106, 306, 259),
                (107, 307, 259),
                (108, 307, 259),
                (109, 307, 259),
                (110, 307, 259),
                (111, 308, 259),
                (112, 308, 259),
                (113, 309, 259),
                (114, 309, 259),
                (115, 310, 259),
                (116, 310, 259),
                (117, 311, 259),
                (118, 311, 259),
                (119, 312, 259),
                (120, 312, 259),
                (121, 313, 259),
                (122, 314, 260),
                (123, 314, 260),
                (124, 315, 260),
                (125, 316, 260),
                (126, 317, 260),
            ],
            [(121, 323, 115), (113, 311, 415)],
        ]

        rd = ResultsDrawer(default_volume_alpha=100, default_point_alpha=150)
        img = rd.drawImageAutocolor(data3d, voxelsize, volumes=masks, points=points)
        # img.show()
        self.assertGreater(img.width, 100)
        self.assertGreater(img.height, 100)

    # def basic_drawer_complex_test(self):
    #     # datap = io3d.read(
    #     #     io3d.datasets.join_path(TEST_DATA_DIR, "PATIENT_DICOM"),
    #     #     dataplus_format=True)
    #     data3d, metadata = io3d.datareader.read(io3d.datasets.join_path(DATA_PATH), dataplus_format=False)
    #     voxelsize = metadata["voxelsize_mm"]
    #     obj = OrganDetection(data3d, voxelsize)
    #     masks = [ obj.getPart(p) for p in ["bones","lungs","kidneys"] ]
    #     bones_stats = obj.analyzeBones()
    #     points = [bones_stats["spine"], bones_stats["hips_start"]]
    #     print(points)
    #     logger.debug(points)
    #
    #     rd = ResultsDrawer(default_volume_alpha=100)
    #     img = rd.drawImageAutocolor(data3d, voxelsize, volumes=masks, points=points)
    #     # img.show()
