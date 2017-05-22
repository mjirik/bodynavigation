#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import argparse
import sys, os

import numpy as np
import scipy
import skimage
import skimage.segmentation
import collections

import io3d
import sed3

import re
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    """ alist.sort(key=natural_keys) sorts in human order """
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true',
        help='run in debug mode')
    parser.add_argument('--all', action='store_true',
        help='analyse all datasets')
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
        help='dataset number to process')
    parser.add_argument('--masks', nargs='+', default=None,
        help='list of mask types to process')
    parser.add_argument('--medianfilter', type=int, default=3,
        help='Size of median filter to use on patient data, default=3')
    args = parser.parse_args()

    # def debug
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    TEST_DATA_DIR = os.path.abspath("./test_data")
    MASKS_PATH = os.path.join(TEST_DATA_DIR, "MASKS_DICOM")
    PATIENT_PATH = os.path.join(TEST_DATA_DIR, "PATIENT_DICOM")
    print("TEST_DATA_DIR:%s" % TEST_DATA_DIR)
    print("MASKS_PATH:%s" % MASKS_PATH)
    print("PATIENT_PATH:%s" % PATIENT_PATH)

    print("MASKS to process: %s" % ("!ALL!" if args.masks is None else str(args.masks)))

    if args.all:
        fl = os.listdir(TEST_DATA_DIR)
        datasets = []
        for f in fl:
            if f not in ["MASKS_DICOM", "PATIENT_DICOM"]:
                datasets.append(f)
        datasets.sort(key=natural_keys)
    elif args.datasets is not None:
        datasets = args.datasets
    else:
        datasets = []
    print("DATASETS to process: %s" % str(datasets))

    print("MedianFilter: %i" % args.medianfilter)

    print("---------------------------------------------------------------------------------------")
    results = [] # aorta, venacava
    for i in datasets:
        dataset_path = os.path.join(TEST_DATA_DIR, "%s" % i)
        if not os.path.exists(dataset_path):
            print("Dataset '%s' in path '%s' not found! Skipping..." % (i, dataset_path))
            continue
        else:
            print("Processing dataset: %s" % i)
        data = collections.OrderedDict({None:collections.OrderedDict({"dataset":i,}),})

        # link this set
        os.system("rm %s" % MASKS_PATH)
        os.system("ln -s %s %s" % (("./%s/MASKS_DICOM/" % i), MASKS_PATH))
        os.system("rm %s" % PATIENT_PATH)
        os.system("ln -s %s %s" % (("./%s/PATIENT_DICOM/" % i), PATIENT_PATH))

        # load patient data
        data3d = io3d.read(PATIENT_PATH, dataplus_format=True)["data3d"]

        # By default rescales values from <-512,511> mode to <-1024,1023> mode.
        # Since io3d uses 512 mode, but ImageJ uses 1024 mode.
        if np.max(data3d) <= 511 and np.min(data3d) >= -512:
            data3d = data3d * 2

        # use median filter
        data3d = scipy.ndimage.filters.median_filter(data3d, args.medianfilter)

        # analyse whole 3d data
        data[None]["median_filter_size"] = args.medianfilter
        data[None]["shape"] = data3d.shape
        data[None]["volume"] = data3d.shape[0]*data3d.shape[1]*data3d.shape[2]
        data[None]["min"] = np.min(data3d)
        data[None]["max"] = np.max(data3d)
        data[None]["median"] = np.median(data3d)
        data[None]["mean"] = np.mean(data3d)
        data[None]["variance"] = np.var(data3d)
        data[None]["standard_deviation"] = np.std(data3d)

        # find out which masks to process
        if args.masks is not None:
            masks_todo = args.masks
        else:
            masks_todo = os.listdir(MASKS_PATH)

        # analyse masks
        for mask in masks_todo:
            testdata_path = os.path.join(MASKS_PATH, mask)
            if not os.path.exists(testdata_path):
                print("Mask '%s' in path '%s' not found! Skipping..." % (mask, testdata_path))
                continue
            else:
                print("Processing mask: %s" % mask)
            mask_inv = io3d.read(testdata_path, dataplus_format=True)["data3d"] == 0
            data_in_mask = np.ma.array(data3d, mask=mask_inv)

            data[mask] = collections.OrderedDict()
            data[mask]["volume"] = np.ma.sum(mask_inv == 0)
            data[mask]["volume_precent"] = data[mask]["volume"]/float(data[None]["volume"])
            data[mask]["min"] = np.ma.min(data_in_mask)
            data[mask]["max"] = np.ma.max(data_in_mask)
            data[mask]["median"] = np.ma.median(data_in_mask)
            data[mask]["mean"] = np.ma.mean(data_in_mask)
            data[mask]["variance"] = np.ma.var(data_in_mask)
            data[mask]["standard_deviation"] = np.ma.std(data_in_mask)

        # save and print data
        results.append([i, data])
        print("==[%s]==" % i)
        for mask in data:
            print("    [%s]" % mask)
            for key in data[mask]:
                print("        %s: %s" % (key, data[mask][key]))

    # print again
    print("---------------------------------------------------------------------------------------")
    for i, data in results:
        print("==[%s]==" % i)
        for mask in data:
            print("    [%s]" % mask)
            for key in data[mask]:
                print("        %s: %s" % (key, data[mask][key]))
    print("---------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
