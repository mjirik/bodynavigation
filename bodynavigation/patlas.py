#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
from __future__ import print_function   # print("text")
from __future__ import division         # 2/3 == 0.666; 2//3 == 0
from __future__ import absolute_import  # 'import submodule2' turns into 'from . import submodule2'
from builtins import range              # replaces range with xrange

import logging
logger = logging.getLogger(__name__)
import traceback

import sys, os
import json
import copy

import numpy as np
import math
import scipy
import scipy.ndimage
import scipy.stats
import skimage.measure
import skimage.transform
import skimage.morphology
import skimage.segmentation
import skimage.feature
import sklearn.mixture

import io3d
import sed3

from .organ_detection import OrganDetection
from .tools import compressArray, decompressArray, NumpyEncoder, readCompoundMask, useDatasetMod
from .transformation import Transformation
from .files import loadDatasetsInfo, joinDatasetPaths

"""
python -m bodynavigation.patlas -h
"""

def ncc(A,B):
    """ normalized cross correlation """
    R = np.corrcoef(A.flatten(), B.flatten())
    return R[0,1]

def ssd(A,B): # TODO - not needed???
    """ Sum of square intensity differences """
    return np.sum((A-B)**2)

def loadDataWithRegPoints(dataset, readydir=None):
    # data3d
    data3d, metadata = io3d.datareader.read(dataset["CT_DATA_PATH"], dataplus_format=False)
    data3d = useDatasetMod(data3d, dataset["MISC"])
    # reg points
    if readydir is not None:
        obj = OrganDetection.fromDirectory(readydir)
    else:
        obj = OrganDetection(data3d, metadata["voxelsize_mm"])

    return data3d, metadata, obj.getRegistrationPoints()

def buildPAtlas(datasets, target_name, readydirs=None, parts=None):
    """
    target_name = "3Dircadb1.1"

    datasets = {
        "3Dircadb1.1":{
            "CT_DATA_PATH":"./3Dircadb1.1/PATIENT_DICOM/",
            "MISC":{ },
            "MASKS":{
                "bodypart1":["./3Dircadb1.1/filepath1","./3Dircadb1.1/filepath2"],
                ...
            }
        },
        ........
        }
    important: datasets must have full paths

    readydirs - path to folder with procesed datasets (folder names are dataset names)
    parts = ["lungs","bones",] - what parts to build patlas for, None is all.
    """
    logger.info("buildPAtlas()")

    # read readydir
    logger.debug("read data in readydir")
    readysets = []
    if readydirs is not None:
        for dirname in next(os.walk(args.readydirs))[1]:
            if dirname in datasets:
                readysets.append(dirname)

    # get target reg points
    logger.debug("get target data3d and registration points")
    t_data3d, t_metadata, t_reg_points = loadDataWithRegPoints( datasets[target_name], \
        readydir= os.path.join(readydirs, target_name) if (target_name in readysets) else None )

    # process train data
    atlas = {}
    for name in datasets:
        logger.debug("Processing dataset: %s" % name)
        s_data3d, s_metadata, s_reg_points = loadDataWithRegPoints( datasets[name], \
        readydir= os.path.join(readydirs, name) if (name in readysets) else None )

        # init registration transformation
        transform = Transformation(s_reg_points, t_reg_points, registration=True)

        # get global weight
        s_data3d = transform.transData(s_data3d)
        w = 1 - ncc(s_data3d,t_data3d)

        # process masks
        for key in datasets[name]["MASKS"]:
            if parts is not None:
                if key not in parts:
                    continue
            if key not in atlas:
                atlas[key] = []

            mask, mask_metadata = readCompoundMask(datasets[name]["MASKS"][key])
            mask = useDatasetMod(mask, datasets[name]["MISC"])
            mask = transform.transData(mask)

            atlas[key].append({
                "w":w, "MASK_COMP":compressArray(mask)
                })
    logger.debug(atlas)

    # build patlas
    PA = {}; PA_info = {"registration_points":t_reg_points, "masks":list(atlas.keys())}
    for key in atlas:
        logger.info("Building PAtlas for: %s" % key)
        PA[key] = np.zeros(t_data3d.shape, dtype=np.float32)

        # calculate raw PA probability
        for i in range(len(atlas[key])):
            mask = decompressArray(atlas[key][i]["MASK_COMP"])
            PA[key] += atlas[key][i]["w"]*mask

        # normalize PA probability
        den = np.sum([ atlas[key][i]["w"] for i in range(len(atlas[key])) ])
        PA[key] /= den

    return PA, PA_info

def savePAtlas(PA, PA_info, path):
    PA_info["data_multiplication"] = 10000.0 # save resolution is 0.01%

    for key in PA:
        logger.info("Saving PAtlas file for '%s'..." % key)
        #PA_uint8 = np.round(PA[key]*255.0).astype(np.int16)
        #ed = sed3.sed3(PA_uint8); ed.show()
        try:
            tmp = (PA[key]*PA_info["data_multiplication"]).astype(np.int16)
            fp = str(os.path.join(path, "%s.dcm" % key)) # IMPORTANT - MUST BE STR CONSTANT (or sitk can throw errors)
            io3d.datawriter.write(tmp, fp, 'dcm', {'voxelsize_mm': PA_info["registration_points"]["spacing"]})
        except:
            traceback.print_exc()

    with open(os.path.join(path, "PA_info.json"), 'w') as fp:
        json.dump(copy.deepcopy(PA_info), fp, encoding="utf-8", cls=NumpyEncoder)

def loadPAtlas(path):
    with open(os.path.join(path, "PA_info.json"), 'r') as fp:
        PA_info = json.load(fp, encoding="utf-8")

    PA = {}
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for fname in onlyfiles:
        fpath = os.path.join(path, fname)
        name, ext = os.path.splitext(fname)
        if ext == ".json": continue
        logger.info("Loading PAtlas from file '%s'" % fname)
        data3d, metadata = io3d.datareader.read(fpath, dataplus_format=False)
        data3d = data3d.astype(np.float32)/PA_info["data_multiplication"] # convert back to percantage
        PA[name] = data3d

    return PA, PA_info

# def segmentation(data3d, PA, PA_info):
#     SEG = {}
#     for key in PA:
#         ## rough segmentation by MAP estimation
#         logger.info("Rough segmentation by MAP estimation...")
#         mean = PA_info[key]["mean"]; std = np.sqrt(PA_info[key]["var"])
#         PA_norm = scipy.stats.norm(mean, std)

#         counts, _ = np.histogram(data3d, bins=range(np.min(data3d), np.max(data3d)+2))
#         data3d_counts = dict(zip(range(np.min(data3d), np.max(data3d)+1), counts))
#         data3d_sum = np.sum(counts); del(counts)

#         Pr_l = PA[key]
#         Pr_I_l = PA_norm.pdf(data3d)

#         # TODO - this part eats +-90s
#         Pr_I = np.zeros(PA[key].shape, dtype=np.float32)
#         for v in range(np.min(data3d), np.max(data3d)+1):
#             Pr_I[data3d==v] = data3d_counts[v]
#         Pr_I = Pr_I / data3d_sum
#         del(data3d_counts, data3d_sum)

#         Pr_l_I = ( Pr_I_l*Pr_l ) / Pr_I
#         C = ( Pr_l_I > 0.7 ).astype(np.uint8) # TODO - this is wrong

#         del(Pr_l, Pr_I_l, Pr_I, Pr_l_I)

#         # use MAP as input for more precise segmentation
#         logger.info("Using MAP as input for more precise segmentation...")
#         # TODO


#         SEG[key] = C.astype(np.uint8) # TODO
#         continue;
#         # # TODO


#         # from pysegbase import pycut


#         # seeds = C
#         # seeds[:,0,0] = 2

#         # gc = pycut.ImageGraphCut(data3d)
#         # gc.set_seeds(seeds)
#         # gc.make_gc()

#         # SEG[key] = gc.segmentation

#         # this implementation of GraphCut eats crazy amount of memory (15GB+)
#         # maybe try directly use pygco library???


#     return SEG


if __name__ == "__main__":
    import argparse

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # input parser
    parser = argparse.ArgumentParser(description="PAtlas")
    parser.add_argument("--buildpa", action="store_true",
            help='')
    parser.add_argument("--segmentation", action="store_true",
            help='')
    parser.add_argument('--datasets', default=io3d.datasets.dataset_path(),
            help='path to dir with raw datasets, default is default io3d.datasets path.')
    parser.add_argument('-po','--patlas_outputdir', default="./PA_patlas",
            help='path to patlas output dir')
    parser.add_argument('-so','--segmentation_outputdir', default="./PA_segmentation",
            help='path to segmentation output dir')
    parser.add_argument('--target', default="3Dircadb1.19",
            help='registration target')
    parser.add_argument('-r','--readydirs', default=None,
            help='path to dir with dirs with processed data3d and masks')
    parser.add_argument('-p','--parts', default=None,
            help='parts to make atlas for, default is all.')
    parser.add_argument("-d", "--debug", action="store_true",
            help='run in debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    print("datasets: ", args.datasets)
    print("patlas_outputdir:", args.patlas_outputdir)
    print("segmentation_outputdir:", args.segmentation_outputdir)
    print("target:", args.target)
    print("readydirs:", args.readydirs)

    # get list of valid datasets and masks
    logger.info("Loading dataset infos")
    datasets = loadDatasetsInfo()

    # update paths to full paths
    logger.info("Updating paths to full paths")
    for name in list(datasets.keys()):
        datasets[name] = joinDatasetPaths(datasets[name], args.datasets)

    # remove datasets that are missing files
    logger.info("Removing dataset infos of missing datasets")
    for name in list(datasets.keys()):
        if not os.path.exists(datasets[name]["CT_DATA_PATH"]):
            del(datasets[name])

    # init output folders
    logger.info("Creating output folders")
    if not os.path.exists(args.patlas_outputdir):
        os.makedirs(args.patlas_outputdir)
    if not os.path.exists(args.segmentation_outputdir):
        os.makedirs(args.segmentation_outputdir)

    # select registration target
    logger.info("Checking registration target")
    if args.target not in datasets:
        logger.error("Target dataset '%s' is not valid dataset!" % args.target)
        sys.exit(2)

    # build patlas
    if args.buildpa:
        logger.info("Building PAtlas")
        parts = None if (args.parts is None) else args.parts.strip().lower().split(",")
        PA, PA_info = buildPAtlas(datasets, args.target, readydirs=args.readydirs, parts=parts)
        #ed = sed3.sed3(PA["liver"]); ed.show()
        savePAtlas(PA, PA_info, args.patlas_outputdir)
    else:
        PA = None; PA_info = None

    # # use patlas for segmentation
    # if args.segmentation:
    #     if PA is None:
    #         PA, PA_info = loadPAtlas(args.patlas_outputdir)
    #     #ed = sed3.sed3(PA["liver"]); ed.show()

    #     data3d, metadata = io3d.datareader.read(all_data[0]["CT_DATA_PATH"], dataplus_format=False)
    #     data3d = useDatasetMod(data3d, all_data[0]["MISC"])
    #     data3d = normalize(data3d, PA["liver"])


    #     SEG = segmentation(data3d, PA, PA_info)
    #     #ed = sed3.sed3(data3d, contour=SEG["liver"]); ed.show()

    #     for key in SEG:
    #         fp = str(os.path.join(output_path, "%s_segmented.dcm" % key))
    #         io3d.datawriter.write(SEG[key], fp, 'dcm', {'voxelsize_mm': (1.0, 1.0, 1.0)})












