#! /usr/bin/env python
# coding: utf-8
#

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

import os

import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

import io3d

# http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.1.zip
TEST_DATA_DIR = "../test_data"

def compute_mask_histograms(data3d, mask, histogram_title="Histogram"): # TODO - remove nan->0 from histograms
    # mask dicom data
    data_in_mask = data3d.copy()
    data_in_mask[mask != 1] = np.ma.masked
    data_outside_mask = data3d.copy()
    data_outside_mask[mask == 1] = np.ma.masked

    # show combined histogram
    bins = np.linspace(np.min(data3d), np.max(data3d), (np.max(data3d)-np.min(data3d))//5)
    plt.hist(data_in_mask[np.isfinite(data_in_mask)], bins, alpha=0.5, label='data in mask')
    plt.hist(data_outside_mask[np.isfinite(data_outside_mask)], bins, alpha=0.5, label='data outside mask')
    plt.legend(loc='upper right')
    plt.title(histogram_title)
    plt.show()

def main():
    logger.setLevel(10)

    # load PATIENT_DICOM data
    logger.info("load PATIENT_DICOM data")
    datap = io3d.read(os.path.join(TEST_DATA_DIR, "PATIENT_DICOM"), dataplus_format=True)
    data3d = datap["data3d"]

    # remove pixel noise
    logger.info("Using median filter")
    data3d = ndimage.filters.median_filter(data3d, 3)

    # body
    logger.info("body")
    mask = io3d.read(os.path.join(TEST_DATA_DIR, "MASKS_DICOM", "skin"), dataplus_format=True)["data3d"] > 0
    compute_mask_histograms(data3d, mask, "Body Histogram")

    # lungs
    logger.info("lungs")
    datap1 = io3d.read(os.path.join(TEST_DATA_DIR, "MASKS_DICOM", "leftlung"), dataplus_format=True)
    datap2 = io3d.read(os.path.join(TEST_DATA_DIR, "MASKS_DICOM", "rightlung"), dataplus_format=True)
    mask = (datap1["data3d"]+datap2["data3d"]) > 0; del(datap1, datap2)
    compute_mask_histograms(data3d, mask, "Lungs Histogram")

    # aorta
    logger.info("aorta")
    mask = io3d.read(os.path.join(TEST_DATA_DIR, "MASKS_DICOM", "artery"), dataplus_format=True)["data3d"] > 0
    compute_mask_histograms(data3d, mask, "Aorta Histogram")

    # venaCava
    logger.info("venaCava")
    mask = io3d.read(os.path.join(TEST_DATA_DIR, "MASKS_DICOM", "venoussystem"), dataplus_format=True)["data3d"] > 0
    compute_mask_histograms(data3d, mask, "venaCava Histogram")


if __name__ == "__main__":
    main()
