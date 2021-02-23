import unittest
import pytest
from bodynavigation.advanced_segmentation import sdf_seg_pipeline
from pathlib import Path


def test_sdf_prepare_data():
    sdf_type = 'surface'
    imshape = 256
    sdf_seg_pipeline.prepare_data(
        n_data=1,
        # skip_h5=True,
        imshape=imshape,
        sdf_type=sdf_type,
        filename_prefix='testfile_', # prevent rewriting the files during test
    )
    assert Path(f"testfile_sdf_{sdf_type}{imshape}.h5").exists()

def test_sdf_training():
    from bodynavigation.advanced_segmentation import sdf_unet256
    sdf_unet256
    assert True

