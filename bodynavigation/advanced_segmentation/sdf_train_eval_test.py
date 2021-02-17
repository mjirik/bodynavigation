import unittest
import pytest
from bodynavigation.advanced_segmentation import sdf_seg_pipeline


def test_sdf_training():
    sdf_seg_pipeline.prepare_data(
        n_data=1,
        # skip_h5=True,
        sdf_type='surface',
        filename_prefix='testfile_', # prevent rewriting the files during test
    )

