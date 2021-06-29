from loguru import logger
import unittest
import pytest
from pathlib import Path
import h5py


# TODO SDF training will be moved into other package
@unittest.skip("not necessary to test on CI")
def test_0_sdf_prepare_data():
    from bodynavigation.advanced_segmentation import sdf_seg_pipeline

    sdf_type = "surface"
    imshape = 256
    filename_prefix = "testfile_"
    expected_fn = Path(f"testfile_sdf_{sdf_type}{imshape}.h5")
    if expected_fn.exists():
        expected_fn.unlink()

    sdf_seg_pipeline.prepare_data(
        n_data=2,
        # skip_h5=True,
        imshape=imshape,
        sdf_type=sdf_type,
        filename_prefix=filename_prefix,  # prevent rewriting the files during test
    )
    assert expected_fn.exists()
    with h5py.File(expected_fn) as h5f:
        logger.debug(h5f.keys())
        assert len(h5f.keys()) == 4


@unittest.skip("not necessary to test on CI")
def test_1_sdf_training():
    from bodynavigation.advanced_segmentation import sdf_unet256

    sdf_type = "surface"
    imshape = 256
    filename_prefix = "testfile_"
    model = sdf_unet256.train(
        sdf_type=sdf_type,
        epochs=3,
        filename_prefix=filename_prefix,
        n_data=2,
        validation_ids=[2],
    )
    # model.fit()
    assert Path(f"{filename_prefix}sdf_unet_{sdf_type}.h5").exists()
