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

import numpy as np
import json
import pandas as pd

import io3d

sys.path.append("..")
import bodynavigation.organ_detection
print("bodynavigation.organ_detection path:", os.path.abspath(bodynavigation.organ_detection.__file__))
from bodynavigation.organ_detection import OrganDetection
from bodynavigation.tools import readCompoundMask, useDatasetMod, NumpyEncoder, naturalSort
from bodynavigation.metrics import compareVolumes
from bodynavigation.files import loadDatasetsInfo, joinDatasetPaths

"""
python batch_organ_detection_analyze_results.py -d -o ./batch_output/ -r ../READY_DIR/
"""

def main():
    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # input parser
    parser = argparse.ArgumentParser(description="Compares segmentation results to masks from datasets. Only compares already segmented data (does not run any segmentation algorithms).")
    parser.add_argument('-i','--datasets', default=io3d.datasets.dataset_path(),
            help='path to dir with raw datasets, default is default io3d.datasets path.')
    parser.add_argument('-o','--outputdir', default="./",
            help='path to output dir')
    parser.add_argument('-on','--outputname', default="output",
            help='name of output files')
    parser.add_argument('-r','--readydirs', default=None,
            help='path to dir with dirs with processed data3d.dcm and masks')
    parser.add_argument('--metrics', default="voe,vd,dice,avgd,rmsd,maxd",
            help='Metrics to use. default: voe,vd,dice,avgd,rmsd,maxd')
    parser.add_argument('--masks', default=None,
            help='Masks to use. Uses all by default.')
    parser.add_argument("-d", "--debug", action="store_true",
            help='run in debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    logging.getLogger("io3d").setLevel(logging.WARNING)

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
    datasets = loadDatasetsInfo()
    for name in datasets:
        datasets[name] = joinDatasetPaths(datasets[name], args.datasets)

    # start comparing masks
    forced_masks = args.masks.strip().lower().split(",") if (args.masks is not None) else args.masks
    output = {}; used_masks = []
    for dirname in sorted(next(os.walk(args.readydirs))[1]):
        if dirname not in datasets:
            continue
        print("Processing:", dirname)

        obj = None; voxelsize_mm = None
        for mask in datasets[dirname]["MASKS"]:
            # check if mask is allowed
            if forced_masks is not None:
                if mask not in forced_masks:
                    continue

            # create paths to masks
            mask_path_dataset = datasets[dirname]["MASKS"][mask]
            mask_path_ready = os.path.join(args.readydirs, dirname, str("%s.dcm" % mask))

            # check if required mask files exist
            if (not np.all([os.path.exists(p) for p in mask_path_dataset+[mask_path_ready,] ])):
                continue
            if mask not in used_masks:
                used_masks.append(mask)
            print("-- comparing mask:", mask)

            # load organ detection
            if obj is None:
                obj = OrganDetection.fromDirectory(os.path.join(args.readydirs, dirname))
                voxelsize_mm = obj.spacing_source

            # read masks
            mask_ready = obj.getPart(mask)
            mask_dataset, _ = readCompoundMask(mask_path_dataset)
            mask_dataset = useDatasetMod(mask_dataset, datasets[dirname]["MISC"])

            # calculate metrics
            if dirname not in output:
                output[dirname] = {}
            output[dirname][mask] = compareVolumes(mask_dataset, mask_ready, voxelsize_mm)

    # save raw output
    output_path = os.path.join(outputdir, str("%s.json" % args.outputname))
    print("Saving output to:", output_path)
    with open(output_path, 'w') as fp:
        json.dump(output, fp, encoding="utf-8", sort_keys=True, indent=4, cls=NumpyEncoder)

    ## Create pandas tables
    print("Create pandas tables...")
    used_metrics = args.metrics.strip().lower().split(",")

    # init columns
    columns = [("","dataset"),]
    for mask in used_masks:
        for met in used_metrics:
            columns.append((mask,met))

    # init dataframe
    df = pd.DataFrame([], columns=columns)

    # write lines
    for dataset in naturalSort(output.keys()):
        line = [dataset,]
        for mask in used_masks:
            # add undefined values
            if mask not in output[dataset]:
                output[dataset][mask] = {}
                for met in used_metrics:
                    output[dataset][mask][met] = None
            # round number
            else:
                for met in used_metrics:
                    output[dataset][mask][met] = round(output[dataset][mask][met], 3)
            # add data to line
            for met in used_metrics:
                line.append(output[dataset][mask][met])
        # add line to table
        tmp = pd.DataFrame([line,], columns=columns)
        df = df.append(tmp)

    # if no data exit
    if df.empty:
        print("Empty Table! Exiting...")
        sys.exit(0)

    # calculate statistics
    df_stats =  df[df.columns[1:]].describe().loc[['50%','std','min','max']]
    df_stats = df_stats.rename(index={'50%': 'median'})
    df_stats = df_stats.round(3)

    # format tables
    if len(used_masks) == 1:
        df.columns = [ c[1] for c in df.columns ]
        df_stats.columns = [ c[1] for c in df_stats.columns ]
    else:
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["",""])
        df_stats.columns = pd.MultiIndex.from_tuples(df_stats.columns, names=["",""])
    df.fillna("-", inplace=True)

    # print tables
    print(df) # index is ignored when saving with index=False
    print(df_stats)

    # write to csv and tex
    df.to_csv(os.path.join(outputdir, str("%s.csv" % args.outputname)), encoding='utf-8', index=False)
    df_stats.to_csv(os.path.join(outputdir, str("%s_stats.csv" % args.outputname)), encoding='utf-8', index=False)

    column_format = "|c|"
    for mask in used_masks:
        column_format += ("c"*len(used_metrics))+"|"
    df.to_latex(os.path.join(outputdir, str("%s.tex" % args.outputname)), encoding='utf-8', index=False, \
        column_format=column_format, multicolumn_format="c|", longtable=False)
    df_stats.to_latex(os.path.join(outputdir, str("%s_stats.tex" % args.outputname)), encoding='utf-8', index=True, \
        column_format=column_format, multicolumn_format="c|", longtable=False)

if __name__ == "__main__":
    main()
