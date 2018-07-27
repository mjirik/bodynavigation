#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
from __future__ import print_function   # print("text")
from __future__ import division         # 2/3 == 0.666; 2//3 == 0
from __future__ import absolute_import  # 'import submodule2' turns into 'from . import submodule2'
from builtins import range              # replaces range with xrange

import logging
logger = logging.getLogger(__name__)

import sys, os, argparse
import traceback

import pkg_resources
import json

import numpy as np

sys.path.append("..")
import bodynavigation.organ_detection
print("bodynavigation.organ_detection path:", os.path.abspath(bodynavigation.organ_detection.__file__))
from bodynavigation.organ_detection import OrganDetection

import io3d
import sed3

"""
python batch_organ_detection_analyze_results.py -d -o ./batch_output/ -r ../READY_DIR/
"""

def diceCoeff(vol1, vol2):
    """ Computes dice coefficient between two binary volumes """
    if (vol1.dtype != np.bool) or (vol2.dtype != np.bool):
        raise Exception("vol1 or vol2 is not np.bool dtype!")
    a = np.sum( vol1[vol2] )
    b = np.sum( vol1 )
    c = np.sum( vol2 )
    return (2*a)/(b+c)

def readCompoundMask(path_list, misc={}):
    # def missing misc variables
    misc["flip_z"] = False if ("flip_z" not in misc) else misc["flip_z"]

    # load masks
    mask, mask_metadata = io3d.datareader.read(path_list[0], dataplus_format=False)
    mask = mask > 0 # to np.bool
    for p in path_list[1:]:
        tmp, _ = io3d.datareader.read(p, dataplus_format=False)
        tmp = tmp > 0 # to np.bool
        mask[tmp] = 1

    # do misc
    if misc["flip_z"]:
        np.flip(mask, axis=0)

    return mask, mask_metadata

def main():
    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # input parser
    parser = argparse.ArgumentParser(description="Compares segmentation results to masks from datasets. Only compares already segmented data (does not run any segmentation algorithms).")
    parser.add_argument('-i','--datasets', default=io3d.datasets.dataset_path(),
            help='path to dir with raw datasets, default is default io3d.datasets path.')
    parser.add_argument('-o','--outputdir', default="./batch_output",
            help='path to output dir')
    parser.add_argument('-r','--readydirs', default=None,
            help='path to dir with dirs with processed data3d.dcm and masks')
    parser.add_argument("-d", "--debug", action="store_true",
            help='run in debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.readydirs is None:
        logger.error("Missing processed data directory path --readydirs")
        sys.exit(2)
    if not os.path.exists(args.datasets) or os.path.isfile(args.datasets):
        logger.error("Invalid data directory path --datasets")
        sys.exit(2)

    outputdir = os.path.abspath(args.outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # print settings
    print("datasets:  %s" % args.datasets)
    print("outputdir: %s" % args.outputdir)
    print("readydirs: %s" % args.readydirs)

    # get list of valid datasets and masks
    datasets = {}
    with pkg_resources.resource_stream("bodynavigation.files", "3Dircadb1.json") as fp:
        datasets.update( json.load(fp, encoding="utf-8") )
    with pkg_resources.resource_stream("bodynavigation.files", "sliver07.json") as fp:
        datasets.update( json.load(fp, encoding="utf-8") )

    # start comparing masks
    output = {}
    for dirname in sorted(next(os.walk(args.readydirs))[1]):
        if dirname not in datasets:
            continue

        print("Processing:", dirname)
        output[dirname] = {}

        # load organ detection
        obj = OrganDetection.fromDirectory(os.path.join(args.readydirs, dirname))

        for mask in datasets[dirname]["MASKS"]:
            mask_path_dataset = [ os.path.join(args.datasets, datasets[dirname]["ROOT_PATH"], \
                part) for part in datasets[dirname]["MASKS"][mask] ]
            mask_path_ready = os.path.join(args.readydirs, dirname, str("%s.dcm" % mask))

            # check if required mask files exist
            if (not np.all([os.path.exists(p) for p in mask_path_dataset+[mask_path_ready,] ])):
                continue
            print("-- comparing mask:", mask)

            # read masks
            mask_ready = obj.getPart(mask)
            mask_dataset, _ = readCompoundMask(mask_path_dataset, datasets[dirname]["MISC"])

            # compare and calculate statistics
            output[dirname][mask] = {}
            output[dirname][mask]["dice"] = diceCoeff(mask_ready, mask_dataset)

    # save raw output
    output_path = os.path.join(outputdir, "output.json")
    print("Saving output to:", output_path)
    with open(output_path, 'w') as fp:
        json.dump(output, fp, encoding="utf-8")

    # TODO - compute statistics, save tables with pandas...

if __name__ == "__main__":
    main()
