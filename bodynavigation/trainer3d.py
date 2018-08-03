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
import tempfile, shutil
import dill as pickle

import numpy as np
import sklearn
import sklearn.mixture

from .tools import toMemMap, concatenateMemMap

class GMMCl(object):
    """
    Modified imtools.ml.gmmcl

    Requires: sklearn>=0.18.0
    """

    def __init__(self, **clspars):
        self.clspars = clspars
        self.cls = {}

    def fit(self, data, target):
        """
        data.dtype = any
        target.dtype = bool OR int
        """
        un = np.unique(target)
        for label in un:
            cli = sklearn.mixture.GaussianMixture(**self.clspars)
            dtl = data[target.reshape(-1)==label]
            cli.fit(dtl)
            self.cls[label] = cli

    def scores(self, x):
        x = np.asarray(x)
        score = {}
        for label in self.cls.keys():
            sc = self.cls[label].score_samples(x)
            score[label] = sc
        return score

    def predict(self, x):
        x = np.asarray(x)
        score_l = {}
        score = []
        for i, label in enumerate(self.cls.keys()):
            score_l[label] = i
            sc = self.cls[label].score_samples(x)
            score.append(sc)
        target_tmp = np.argmax(score, 0)
        return self._relabel(target_tmp, score_l)

    def _relabel(self, target, new_keys):
        out = np.zeros(target.shape, dtype=target.dtype)
        for label, i in new_keys.iteritems():
            out[target==i] = label
        return out

class Trainer3D():
    """
    Refractored and improved version of imtools.trainer3d

    If feature_function() needs resized data (to same shape), do that before using this class.
    """

    def __init__(self, feature_function=None, train_nth=50, memmap=True): # TODO - test how much does memmap lower MEM usage
        """
        feature_function = feature_function(**fv_kwargs)
            - should return numpy vector for every voxel in data3d
        """
        # Classifier
        self.cl = GMMCl(n_components=6)
        # self.cl = imtools.ml.gmmcl.GMMCl(n_components=6)
        # self.cl = sklearn.naive_bayes.GaussianNB()
        # self.cl = sklearn.mixture.GMM()
        # self.cl = sklearn.tree.DecisionTreeClassifier()
        if feature_function is None:
            feature_function = self.defaultFeatureFunction
        self.feature_function = feature_function

        # training
        self.train_data = None
        self.train_output = None
        self.train_nth = train_nth

        # memmap
        self.memmap = memmap
        if self.memmap:
            self.tempdir = tempfile.mkdtemp(prefix="trainer3d_")
        else:
            self.tempdir = None

    def __del__(self):
        """ Decontructor """

        # imports are set to None or deleted on app exit
        try:
            shutil.get_archive_formats()
        except:
            import shutil

        # remove tempdir
        if self.tempdir is not None:
            shutil.rmtree(self.tempdir)

    def save(self, file_path="trainer3d.pickle"):
        sv = { "cl": self.cl, }
        with open(file_path, "wb") as fp:
            pickle.dump(sv, fp) # TODO - is this safe?

    def load(self, file_path="trainer3d.pickle"):
        """ Load pretrained Classifier """
        with open(file_path, "rb") as fp:
            sv = pickle.load(fp)
        self.cl = sv["cl"]

    @classmethod
    def fromFile(cls, file_path):
        obj = cls()
        obj.load(file_path)
        return obj

    ###########################
    # Training and Prediction #
    ###########################

    @staticmethod
    def defaultFeatureFunction(**kwargs):
        """ Default feature_function """
        fv_list = []

        if "data3d" in kwargs: # intensity model
            fv_list.append( kwargs["data3d"].reshape(-1, 1) )
        if "patlas" in kwargs: # patlas
            fv_list.append( kwargs["patlas"].reshape(-1, 1) )
        for key in kwargs: # distances to parts
            if key.startswith("dist_"):
                fv_list.append( kwargs[key].reshape(-1, 1) )

        if len(fv_list) == 0:
            raise Exception("Empty feature vector!")
        fv = np.concatenate(fv_list, 1)
        return fv

    def addTrainData(self, output, **fv_kwargs):
        """ output - targeted output of predict() """
        # get feature vectors
        fv = self.feature_function(**fv_kwargs)

        # use only fraction of data to save on memory
        data = fv[::self.train_nth]
        output = np.reshape(output, [-1, 1])[::self.train_nth]

        # save train data
        if self.train_data is None:
            if self.memmap:
                self.train_data = toMemMap(data, os.path.join(self.tempdir, "train_data.arr"))
                self.train_output = toMemMap(output, os.path.join(self.tempdir, "train_output.arr"))
            else:
                self.train_data = data
                self.train_output = output
        else:
            if self.memmap:
                self.train_data = concatenateMemMap(self.train_data, data)
                self.train_output = concatenateMemMap(self.train_output, output)
            else:
                self.train_data = np.concatenate([self.train_data, data], 0)
                self.train_output = np.concatenate([self.train_output, output], 0)

    def fit(self):
        logger.debug("train_data bytesize: %s" % self.train_data.nbytes)
        logger.debug("train_output bytesize: %s" % self.train_output.nbytes)
        self.cl.fit(self.train_data, self.train_output)

    def predict(self, shape, **fv_kwargs):
        """ shape - expected shape of output data """
        fv = self.feature_function(**fv_kwargs)
        return self.cl.predict(fv).reshape(shape)

    def predictW(self, shape, weight, label0=0, label1=1, **fv_kwargs): # TODO - untested
        """
        segmentation with weight factor
        shape - expected shape of output data
        """
        scores = self.scores(shape, **fv_kwargs)
        return scores[label1] > (weight * scores[label0])

    def scores(self, shape, **fv_kwargs): # TODO - untested
        """ shape - expected shape of output data """
        fv = self.feature_function(**fv_kwargs)
        scores = self.cl.scores(fv)
        for key in scores:
            scores[key] = scores[key].reshape(shape)
        return scores


if __name__ == "__main__":
    import argparse
    import scipy
    import io3d
    import sed3

    from .files import loadDatasetsInfo, joinDatasetPaths, addDatasetRegPoints, getDefaultPAtlas
    from .tools import readCompoundMask, useDatasetMod
    from .transformation import Transformation
    from .patlas import loadPAtlas

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # input parser
    parser = argparse.ArgumentParser(description="Trainer3D")
    parser.add_argument('--datasets', default=io3d.datasets.dataset_path(),
            help='path to dir with raw datasets, default is default io3d.datasets path.')
    parser.add_argument('--patlas', default=None,
            help='path to custom patlas, if not defined uses default patlas.')
    parser.add_argument('-o','--outputdir', default="./Trainer3D_output",
            help='path to output dir')
    parser.add_argument('-r','--readydirs', default=None,
            help='path to dir with dirs with processed data3d and masks')
    parser.add_argument('-p','--parts', default=None,
            help='parts to train for, default is all.')
    parser.add_argument("-d", "--debug", action="store_true",
            help='run in debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    print("datasets: ", args.datasets)
    print("patlas:", args.patlas)
    print("outputdir:", args.outputdir)
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

    # init output folder
    logger.info("Creating output folder")
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # update datasets infos with registration points
    logger.info("update datasets infos with registration points")
    datasets = addDatasetRegPoints(datasets, args.readydirs)

    # patlas
    patlas_path = args.patlas
    if patlas_path is None:
        patlas_path = tempfile.mkdtemp(prefix="trainer3d_patlas_")
        getDefaultPAtlas(patlas_path)
    PA, PA_info = loadPAtlas(patlas_path)

    #####################
    # train classifiers #
    #####################
    logger.info("Training classifiers")
    parts = list(PA.keys()) if (args.parts is None) else args.parts.strip().lower().split(",")
    for part in parts:
        if part not in list(PA.keys()):
            print("Skipping invalid part name:", part)
            continue
        print("Training classifier for part:", part)
        ol = Trainer3D(); data_i = 0

        # add training data
        for name in datasets:
            if part not in datasets[name]["MASKS"]:
                continue

            # classification target
            mask, _ = readCompoundMask(datasets[name]["MASKS"][part])
            mask = useDatasetMod(mask, datasets[name]["MISC"])

            # data3d
            data3d, _ = io3d.datareader.read(datasets[name]["CT_DATA_PATH"], dataplus_format=False)
            data3d = useDatasetMod(data3d, datasets[name]["MISC"])
            data3d = scipy.ndimage.filters.median_filter(data3d, 3) # remove noise

            # patlas
            transform = Transformation(PA_info["registration_points"], \
                datasets[name]["REG_POINTS"], registration=True)
            patlas = transform.transData(PA[part])
            #ed = sed3.sed3(patlas); ed.show()

            # add train data to classifier
            ol.addTrainData(mask, data3d=data3d, patlas=patlas)
            data_i += 1

        # test if we added any data at all
        if data_i == 0:
            print("No masks for part '%s'! Nothing trained!" % part)
            continue

        # train and save classifier
        ol.fit()
        ol.save(os.path.join(args.outputdir, str("%s.pickle" % part)))
        del(ol)


# TODO - remove
# def localization_fv_navigation_intensity(data3d, voxelsize_mm):
#         fv = []
#         f0 = scipy.ndimage.filters.gaussian_filter(data3d, sigma=3).reshape(-1, 1)
#         import bodynavigation as bn
#         ss = bn.BodyNavigation(data3d, voxelsize_mm)
#         fd1 = ss.dist_to_lungs().reshape(-1, 1)
#         fd2 = ss.dist_to_spine().reshape(-1, 1)
#         fd3 = ss.dist_sagittal().reshape(-1, 1)
#         fd4 = ss.dist_coronal().reshape(-1, 1)
#         fd5 = ss.dist_axial().reshape(-1, 1)
#         fd6 = ss.dist_to_surface().reshape(-1, 1)
#         fd7 = ss.dist_diaphragm().reshape(-1, 1)
#         fv = np.concatenate([
#                 f0,
#                 fd1, fd2, fd3, fd4, fd5, fd6, fd7,
#             ], 1)
#         return fv
