#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
import sys

import numpy as np
from nose.plugins.attrib import attr

# from lisa import organ_segmentation
# import pysegbase.dcmreaddata as dcmr
# import lisa.data


# nosetests tests/organ_segmentation_test.py:OrganSegmentationTest.test_create_iparams # noqa

class BodyNavigationOnGeneratedDataTest(unittest.TestCase):
    interactiveTest = False
    verbose = False

    def generate_data(self):

        img3d = (np.random.rand(30, 30, 30)*10).astype(np.int16)
        seeds = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation[10:25, 4:24, 2:16] = 1
        img3d = img3d + segmentation*20
        seeds[12:18, 9:16, 3:6] = 1
        seeds[19:22, 21:27, 19:21] = 2

        voxelsize_mm = [5, 5, 5]
        metadata = {'voxelsize_mm': voxelsize_mm}
        return img3d, metadata, seeds, segmentation

    # @unittest.skipIf(not interactiveTest, "interactive test")
    @attr("interactive")
    def test_viewer_seeds(self):

        try:
            from pysegbase.seed_editor_qt import QTSeedEditor
        except:
            print("Deprecated of pyseg_base as submodule")
            from seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication
        import numpy as np
        img3d = (np.random.rand(30, 30, 30)*10).astype(np.int16)
        seeds = (np.zeros(img3d.shape)).astype(np.int8)
        seeds[3:6, 12:18, 9:16] = 1
        seeds[3:6, 19:22, 21:27] = 2
        # , QMainWindow
        app = QApplication(sys.argv)
        pyed = QTSeedEditor(img3d, seeds=seeds)
        pyed.exec_()

        # deletemask = pyed.getSeeds()
        # import pdb; pdb.set_trace()

        # pyed = QTSeedEditor(deletemask, mode='draw')
        # pyed.exec_()

        app.exit()
    # @unittest.skip("demonstrating skipping")

    # @attr("interactive")
    # def test_whole_organ_segmentation_interactive(self):
    #     """
    #     Interactive test uses dicom data for segmentation
    #     """
    #     dcmdir = os.path.join(
    #         lisa.data.sample_data_path(),
    #         'matlab/examples/sample_data/DICOM/digest_article/'
    #     )
    #         # path_to_script,
    #         # './../sample_data/matlab/examples/sample_data/DICOM/digest_article/') # noqa

if __name__ == "__main__":
    unittest.main()
