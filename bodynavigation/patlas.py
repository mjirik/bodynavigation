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

import json
import pkg_resources

import io3d
import sed3

# run with: "python -m bodynavigation.patlas -h"
from .tools import compressArray, decompressArray, NumpyEncoder
from .transformation import TransformationNone, Transformation


# class ProbabilisticAtlas(object):
#     pass

#     """
#     Input: data3d, spacing + abdomen mask

#     """

#     @classmethod
#     def buildPAtlas(cls, data_paths, output_path):
#         pass

    # @classmethod
    # def _getLiver(cls): # TODO
    #     pass

    # @classmethod
    # def _getSpleen(cls): # TODO
    #     pass

    # @classmethod
    # def _getKidneys(cls): # TODO
    #     pass



def normalize(data, target=None): # TODO - # TransformationNone, Transformation
    if target is None:
        next_power_of_2 = lambda x: int(1 if x == 0 else 2**math.ceil(math.log(x,2)))
        new_shape = tuple([ next_power_of_2(s) for s in data.shape ])
    else:
        new_shape = target.shape

    out = skimage.transform.resize(
        data, new_shape, order=3, mode="reflect", clip=True, preserve_range=True
        )
    if data.dtype in [np.bool,np.int,np.uint]:
        out = np.round(out)
    out = out.astype(data.dtype)

    return out

def ncc(A,B):
    """ normalized cross correlation """
    R = np.corrcoef(A.flatten(), B.flatten())
    return R[0,1]

def ssd(A,B):
    """ Sum of square intensity differences """
    return np.sum((A-B)**2)

def spatialDivision(M,N=1):
    """
    Retrns list of views of subspaces of original array
    N = (2^k)^3 is number of created subspaces where 2^k is number of subspaces along one axis
    """
    N_axis = int(math.ceil(N**(1/3))) # 2^k

    # expand shape, so we are working as if it is 2^k on every axis
    next_power_of_2 = lambda x: int(1 if x == 0 else 2**math.ceil(math.log(x,2)))
    shape = [ next_power_of_2(s) for s in M.shape ]
    U_shape = [ shape[0]//N_axis, shape[1]//N_axis, shape[2]//N_axis ]

    # split into subspaces
    subspaces = []
    for zi in range(0,N):
        for yi in range(0,N):
            for xi in range(0,N):
                subspaces.append(M[ \
                    (zi*U_shape[0]):((zi+1)*U_shape[0]), \
                    (yi*U_shape[1]):((yi+1)*U_shape[1]), \
                    (xi*U_shape[2]):((xi+1)*U_shape[2]) \
                    ])

    return subspaces

def readCompoundMask(path_list):
    mask, mask_metadata = io3d.datareader.read(path_list[0], dataplus_format=False)
    mask = mask > 0 # to np.bool
    for p in path_list[1:]:
        tmp, _ = io3d.datareader.read(p, dataplus_format=False)
        tmp = tmp > 0 # to np.bool
        mask[tmp] = 1
    return mask, mask_metadata

def buildPAtlas(target_data, train_data, N=4**3):
    """
    target_data["CT_DATA_PATH"] - filesystem path to dicom with target image
    train_data = [{
        "CT_DATA_PATH": path to data3d,
        "liver": path to mask (dtype=np.bool),
        ...
        },...]
    """
    logger.info("Starting build of PAtlas...")

    # prepare target image
    I, metadata = io3d.datareader.read(target_data["CT_DATA_PATH"], dataplus_format=False)
    #I_vs = metadata["voxelsize_mm"]
    I = normalize(I)
    I_Uj = spatialDivision(I,N)

    # process train data
    Atlas = {}
    for i in range(len(train_data)):
        logger.info("Processing: %s" % train_data[i]["CT_DATA_PATH"])
        data3d, metadata = io3d.datareader.read(train_data[i]["CT_DATA_PATH"], dataplus_format=False)
        #data3d_vs = metadata["voxelsize_mm"]
        data3d = normalize(data3d,I)
        data3d_Uj = spatialDivision(data3d,N)

        # get global weight
        w = 1 - ncc(data3d,I)

        # get local weights
        wj = [ ( 1 - ssd(data3d_Uj[j], I_Uj[j]) ) for j in range(len(I_Uj)) ]

        # process masks
        for key in train_data[i]:
            if key == "CT_DATA_PATH": continue
            if key not in Atlas: Atlas[key] = []

            mask, mask_metadata = readCompoundMask(train_data[i][key])
            mask = normalize(mask,I) # TODO - need to use identical transform as on data3d
            X = data3d[mask >= 1] # values of masked data3d

            Atlas[key].append({
                "w":w, "wj":wj, "MASK_COMP":compressArray(mask),
                "mean":np.mean(X), "var":np.var(X)
                })

            del(mask); del(X)

        del(data3d_Uj); del(data3d)

    PA = {}; PA_stats = {}
    for key in Atlas:
        logger.info("Building PAtlas for '%s'..." % key)
        PA[key] = np.zeros(I.shape, dtype=np.float32)
        PA_Uj = spatialDivision(PA[key],N)

        for i in range(len(Atlas[key])):
            # calculate raw PA probability
            mask = decompressArray(Atlas[key][i]["MASK_COMP"])
            mask_Uj = spatialDivision(mask,N)
            for j in range(len(PA_Uj)):
                PA_Uj[j] += Atlas[key][i]["w"]*Atlas[key][i]["wj"][j]*mask_Uj[j]
            del(mask)

        # normalize with "sum_i ( wi*wij )" for every subspace
        den_j = np.zeros((len(PA_Uj), ), dtype=np.float32)
        for i in range(len(Atlas[key])):
            den_j = den_j + (Atlas[key][i]["w"]*np.asarray(Atlas[key][i]["wj"]))
        for j in range(len(PA_Uj)):
            PA_Uj[j] /= den_j[j]

        # estimate mean and variance intensity
        mean_i = []; var_i = []
        for i in range(len(Atlas[key])):
            mean_i.append(Atlas[key][i]["mean"])
            var_i.append(Atlas[key][i]["var"])
        PA_stats[key] = {"mean":np.mean(mean_i), "var":np.mean(var_i)}

    return PA, PA_stats

def savePAtlas(PA, PA_stats, path):
    for key in PA:
        logger.info("Saving PAtlas file for '%s'..." % key)
        #PA_uint8 = np.round(PA[key]*255.0).astype(np.int16)
        #ed = sed3.sed3(PA_uint8); ed.show()
        try:
            tmp = (PA[key]*10000.0).astype(np.int16) # save resolution is 0.01%
            fp = str(os.path.join(path, "%s.dcm" % key)) # IMPORTANT - MUST BE STR CONSTANT (or sitk can throw errors)
            io3d.datawriter.write(tmp, fp, 'dcm', {'voxelsize_mm': (1.0, 1.0, 1.0)})
        except:
            traceback.print_exc()
    with open(os.path.join(path, "PA_stats.json"), 'w') as fp:
        json.dump(PA_stats, fp, encoding="utf-8", cls=NumpyEncoder)

def loadPAtlas(path):
    PA = {}
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for fname in onlyfiles:
        fpath = os.path.join(path, fname)
        name, ext = os.path.splitext(fname)
        if ext == ".json": continue
        logger.info("Loading PAtlas from file '%s'" % fname)
        data3d, metadata = io3d.datareader.read(fpath, dataplus_format=False)
        data3d = data3d.astype(np.float32)/10000.0 # convert back to percantage
        PA[name] = data3d

    with open(os.path.join(path, "PA_stats.json"), 'r') as fp:
        PA_stats = json.load(fp, encoding="utf-8")

    return PA, PA_stats

def segmentation(data3d, PA, PA_stats):
    SEG = {}
    for key in PA:
        ## rough segmentation by MAP estimation
        logger.info("Rough segmentation by MAP estimation...")
        mean = PA_stats[key]["mean"]; std = np.sqrt(PA_stats[key]["var"])
        PA_norm = scipy.stats.norm(mean, std)

        counts, _ = np.histogram(data3d, bins=range(np.min(data3d), np.max(data3d)+2))
        data3d_counts = dict(zip(range(np.min(data3d), np.max(data3d)+1), counts))
        data3d_sum = np.sum(counts); del(counts)

        Pr_l = PA[key]
        Pr_I_l = PA_norm.pdf(data3d)

        # TODO - this part eats +-90s
        Pr_I = np.zeros(PA[key].shape, dtype=np.float32)
        for v in range(np.min(data3d), np.max(data3d)+1):
            Pr_I[data3d==v] = data3d_counts[v]
        Pr_I = Pr_I / data3d_sum
        del(data3d_counts, data3d_sum)

        Pr_l_I = ( Pr_I_l*Pr_l ) / Pr_I
        C = ( Pr_l_I > 0.7 ).astype(np.uint8) # TODO - this is wrong

        del(Pr_l, Pr_I_l, Pr_I, Pr_l_I)

        # use MAP as input for more precise segmentation
        logger.info("Using MAP as input for more precise segmentation...")
        # TODO


        SEG[key] = C.astype(np.uint8) # TODO
        continue;
        # # TODO


        # from pysegbase import pycut


        # seeds = C
        # seeds[:,0,0] = 2

        # gc = pycut.ImageGraphCut(data3d)
        # gc.set_seeds(seeds)
        # gc.make_gc()

        # SEG[key] = gc.segmentation

        # this implementation of GraphCut eats crazy amount of memory (15GB+)
        # maybe try directly use pygco library???


    return SEG


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
    parser.add_argument("--N", default=4**3, type=int,
            help='')
    parser.add_argument("-d", "--debug", action="store_true",
            help='run in debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    all_data = []

    #### sliver07 DATA

    data_root = "/home/jirka642/Programming/_Data/DP/sliver07/PREPARED"
    #for i in [1,2,3]:
    for i in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
        all_data.append({
            "CT_DATA_PATH": os.path.join(data_root, "silver07-orig%s" % str(i).zfill(3), "silver07-orig%s.dcm" % str(i).zfill(3) ),
            "liver": [os.path.join(data_root, "silver07-seg%s" % str(i).zfill(3), "silver07-seg%s.dcm" % str(i).zfill(3) ),]
            })

    #### 3Dircadb1 DATA

    data_root = "/home/jirka642/Programming/_Data/DP/3Dircadb1"
    with pkg_resources.resource_stream("bodynavigation.files", "3Dircadb1.json") as fp:
        ircad_info = json.load(fp, encoding="utf-8")

    #for k in [1,2,4]:
    for k in [1,2,4,5,6,7,8,9,10,11,13,14,15,16,17,18]: # 19
        k = str(k)
        tmp = {
            "CT_DATA_PATH": os.path.join(data_root, ircad_info[k]["ROOT_PATH"], ircad_info[k]["CT_DATA_PATH"])
        }
        for mk in ircad_info[k]:
            if mk in ["CT_DATA_PATH", "ROOT_PATH"]: continue
            tmp[mk] = []
            for p in ircad_info[k][mk]:
                tmp[mk].append( os.path.join(data_root, ircad_info[k]["ROOT_PATH"], p) )
        all_data.append(tmp)

    ####

    patlas_path = "/home/jirka642/Programming/Sources/bodynavigation/PATLAS/"
    output_path = "/home/jirka642/Programming/Sources/bodynavigation/TMP/"

    target_data = {
        "CT_DATA_PATH":all_data[0]["CT_DATA_PATH"]
        }
    train_data = all_data[1:] # remove target image

    ####

    PA = None
    if args.buildpa:
        PA, PA_stats = buildPAtlas(target_data, train_data, N=args.N)
        #ed = sed3.sed3(PA["liver"]); ed.show()
        savePAtlas(PA, PA_stats, patlas_path)

    if args.segmentation:
        if PA is None:
            PA, PA_stats = loadPAtlas(patlas_path)
        #ed = sed3.sed3(PA["liver"]); ed.show()

        data3d, metadata = io3d.datareader.read(all_data[0]["CT_DATA_PATH"], dataplus_format=False)
        data3d = normalize(data3d, PA["liver"])


        SEG = segmentation(data3d, PA, PA_stats)
        #ed = sed3.sed3(data3d, contour=SEG["liver"]); ed.show()

        for key in SEG:
            fp = str(os.path.join(output_path, "%s_segmented.dcm" % key))
            io3d.datawriter.write(SEG[key], fp, 'dcm', {'voxelsize_mm': (1.0, 1.0, 1.0)})











