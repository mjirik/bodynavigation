#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
from __future__ import print_function   # print("text")
from __future__ import division         # 2/3 == 0.666; 2//3 == 0
from __future__ import absolute_import  # 'import submodule2' turns into 'from . import submodule2'
from builtins import range              # replaces range with xrange

import os
import pkg_resources
import json
import zipfile

DATASETS_FILENAMES = ["3Dircadb1.json", "3Dircadb2.json", "sliver07.json"]
def loadDatasetsInfo():
    datasets = {}
    for fn in DATASETS_FILENAMES:
        with pkg_resources.resource_stream("bodynavigation.files", fn) as fp:
            datasets.update( json.load(fp, encoding="utf-8") )
    return datasets

def joinDatasetPaths(dataset, root_path=""):
    root_path = os.path.join(root_path, dataset["ROOT_PATH"]); dataset["ROOT_PATH"] = ""
    dataset["CT_DATA_PATH"] = os.path.join(root_path, dataset["CT_DATA_PATH"])
    for mask in dataset["MASKS"]:
        dataset["MASKS"][mask] = [ os.path.join(root_path, pp) for pp in dataset["MASKS"][mask] ]
    return dataset

def getDefaultPAtlas(path):
    """ Extracts default PAtlas to path """
    with pkg_resources.resource_stream("bodynavigation.files", "default_patlas.zip") as fp:
        zip_ref = zipfile.ZipFile(fp)
        zip_ref.extractall(path)
        zip_ref.close()
