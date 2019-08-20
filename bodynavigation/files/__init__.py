#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
from __future__ import print_function  # print("text")
from __future__ import division  # 2/3 == 0.666; 2//3 == 0
from __future__ import (
    absolute_import,
)  # 'import submodule2' turns into 'from . import submodule2'
from builtins import range  # replaces range with xrange

import os
import pkg_resources
import json
import zipfile

# Datasets

DATASETS_FILENAMES = ["3Dircadb1.json", "3Dircadb2.json", "sliver07.json"]


def loadDatasetsInfo():
    datasets = {}
    for fn in DATASETS_FILENAMES:
        with pkg_resources.resource_stream("bodynavigation.files", fn) as fp:
            datasets.update(json.load(fp, encoding="utf-8"))
    return datasets


def joinDatasetPaths(dataset, root_path=""):
    """ Also removes ROOT_PATH to prevent confusion """
    root_path = os.path.join(root_path, dataset["ROOT_PATH"])
    del dataset["ROOT_PATH"]
    dataset["CT_DATA_PATH"] = os.path.join(root_path, dataset["CT_DATA_PATH"])
    for mask in dataset["MASKS"]:
        dataset["MASKS"][mask] = [
            os.path.join(root_path, pp) for pp in dataset["MASKS"][mask]
        ]
    return dataset


def addDatasetRegPoints(
    datasets, readydirs=None
):  # TODO - do I need this anywhere but patlas???
    # imports must be here to prevent possible cyclic imports of OrganDetection
    import io3d
    from ..organ_detection import OrganDetection
    from ..tools import useDatasetMod

    readysets = []
    if readydirs is not None:
        for dirname in next(os.walk(readydirs))[1]:
            if dirname in datasets:
                readysets.append(dirname)
    for name in datasets:
        dataset = datasets[name]
        # test if reg points are already defined
        if "REG_POINTS" in dataset:
            if dataset["REG_POINTS"] is not None:
                continue
        # get reg points
        readydir = os.path.join(readydirs, name) if (name in readysets) else None
        if readydir is not None:
            obj = OrganDetection.fromDirectory(readydir)
        else:
            data3d, metadata = io3d.datareader.read(
                dataset["CT_DATA_PATH"], dataplus_format=False
            )
            data3d = useDatasetMod(data3d, dataset["MISC"])
            obj = OrganDetection(data3d, metadata["voxelsize_mm"])
        datasets[name]["REG_POINTS"] = obj.getRegistrationPoints()

    return datasets


# PAtlas


def getDefaultPAtlas(path):
    """ Extracts default PAtlas to path """
    with pkg_resources.resource_stream(
        "bodynavigation.files", "default_patlas.zip"
    ) as fp:
        zip_ref = zipfile.ZipFile(fp)
        zip_ref.extractall(path)
        zip_ref.close()


# Classifier


def getDefaultClassifier(path):
    """ Extracts default Classifier to path """
    with pkg_resources.resource_stream(
        "bodynavigation.files", "default_classifier.zip"
    ) as fp:
        zip_ref = zipfile.ZipFile(fp)
        zip_ref.extractall(path)
        zip_ref.close()
