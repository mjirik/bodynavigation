import pytest
import io3d
import sed3
from loguru import logger
import numpy as np
import pandas as pd
from . import kidney_segmentation

# # On my harddrive, the data3d is stored in .mhd file so i had to change the dataset path pattern
# io3d.datasets.DATASET_PATH_STRUCTURE["pilsen_pigs"]={
#     '_': 'medical/orig/pilsen_pigs/Tx{id:03d}D_{subtype}/MASKS_DICOM/{data_type}/', # masks
#     'data3d': 'medical/orig/pilsen_pigs/Tx{id:03d}D_{subtype}/PATIENT_DICOM/Tx{id:03d}D_{subtype}.mhd' # intensity data
# }

def test_trenovani():
    kidney_segmentation.vytiskni("ahoj")
    dp = io3d.read_dataset("3Dircadb1", 'data3d', 1)
    # ed = sed3.sed3(dp['data3d'])
    # ed.show()
    sed3.show_slices(dp['data3d'], slice_number=24)


def test_trainer():
    ol=kidney_segmentation.train([30, 30])
    kidney_segmentation.klasification(30, ol)


def test_train_and_save():
    ol = kidney_segmentation.train([27], show=True, organ_label="right_kidney")
    import joblib
    joblib.dump(ol, 'ol.joblib')


def test_train_load():
    import joblib
    ol=joblib.load('ol.joblib')
    print(ol)
    kidney_segmentation.klasification(29, ol)


def test_check_path_to_pilsen_pigs():
    pth = io3d.datasets.get_dataset_path("pilsen_pigs", 'data3d', 27)
    logger.debug(pth)
