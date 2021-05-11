import io3d
import os
import os.path
import pytest
import os.path as op
import sys
import matplotlib.pyplot as plt
import scipy
import glob
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path("~/projects/imtools").expanduser()))
import sklearn.neural_network
import imtools.trainer3d
import imtools.datasets
import imtools.ml
import io3d
import sed3
from loguru import logger
from sklearn.svm import SVC
import bodynavigation
import imma
# logger.disable("io3d")
logger.remove()
logger.add(sys.stderr, level='INFO')

def vytiskni(text):
    print(text)

# Tady by měly být funkce, které máte teď v notebooku

# Define feature function
def externfv(data3d, voxelsize_mm):        # scale
    f0 = scipy.ndimage.filters.gaussian_filter(data3d, sigma=0)
    f1 = scipy.ndimage.filters.gaussian_filter(data3d, sigma=1)- f0 #hranový.
    ss = bodynavigation.body_navigation.BodyNavigation(data3d, voxelsize_mm)
    f3 = ss.dist_sagittal()
    f4 = ss.dist_coronal()
    f5= ss.dist_to_diaphragm_axial() # distance from heart
    #dist saggital, coronall
    fv = np.concatenate([
        f0.reshape(-1, 1), f1.reshape(-1, 1), f3.reshape(-1, 1),
        f4.reshape(-1, 1),
        f5.reshape(-1, 1)
    ], 1)
    np.save('Abc.npy', fv)
    #np.load
    return fv
# Trainer function

def train(patients_IDs, show=False, organ_label="RightKidney", do_balance=True): # a - first CT patient, b - last CT patient. We need existence of all patients between this 2 numbers.
    ol = imtools.trainer3d.Trainer3D()
    ol.working_voxelsize_mm = [3, 3, 3]
    # select feature function
    ol.feature_function = externfv
    # select classifierS
    #ol.cl=SVC(kernel='linear', class_weight='balanced', probability=True)
    ol.cl = imtools.ml.gmmcl.GMMCl()
    #ol.cl= sklearn.neural_network.MLPClassifier()
    #patient = range(first_patient, last_patient + 1)
    #import sklearn.tree
    #ol.cl = sklearn.tree.DecisionTreeClassifier()
    ol.cl.cls = {0: sklearn.mixture.GaussianMixture(n_components=3), 1: sklearn.mixture.GaussianMixture(n_components=1)}
    for z in patients_IDs:
        i = z
        datap = io3d.datasets.read_dataset("pilsen_pigs", 'data3d', i)
        datap['data3d'][datap['data3d']<-200]=-200
        datap_liver = io3d.datasets.read_dataset("pilsen_pigs", organ_label, i)
        # print('datap', datap)

        # print('datap_liver', datap_liver)
        z=z+3 # mezi 26 a 30 4 patieny
        mask_kidney = datap_liver['data3d'] > 0

        #sed3.ipy_show_slices(mask_kidney)
        if show:
            ad=sed3.sed3(datap['data3d'], contour=mask_kidney)
            ad.set_window(40,400)
            ad.show()
        data3d = datap["data3d"]
        MASK = (mask_kidney).astype(np.uint8)
        logger.debug(np.unique(MASK, return_counts=True))

        nth = 1 # save all voxels
        ol.add_train_data(data3d, MASK, voxelsize_mm=datap["voxelsize_mm"], nth=nth)
        if do_balance:
            data, target = balance_data(ol.data, ol.target)
            ol.data = data
            ol.target = target

    np.save('data.npy', ol.data)
    np.save('target.npy', ol.target)
    ol.fit()
    if show:
        plt.hist(data3d.reshape(-1, 1), bins=30)
        plt
    return ol

def klasification(test_patient, ol, first_slice=50, last_slice=900, show=True):
    # one = list(imtools.datasets.sliver_reader("*000.mhd", read_seg=True))[0]
    # numeric_label, vs_mm, oname, orig_data, rname, ref_data = one
    i = test_patient
    datap = io3d.datasets.read_dataset("pilsen_pigs", 'data3d', i)
    datap['data3d'][datap['data3d'] < -2000] = -1000
    crop=datap["data3d"][first_slice:last_slice:1] # purt of patient
    fit = ol.predict(crop, voxelsize_mm=datap["voxelsize_mm"])
    if show:
        ad=sed3.sed3(crop, contour=fit)
        #ad = sed3.sed3(fit)
        ad.set_window(40, 400)
        ad.show()
    #plt.imshow(crop[125], cmap='gray', clim=[-200, 200])
    #plt.contour(fit[125])
    #plt.show()


def balance_data(X, y):
    import sklearn.utils

    cls, cts = np.unique(y, return_counts=True)
    mn = np.min(cts)
    balanced_data = []
    balanced_target = []
    for cl in cls:
        x_i = sklearn.utils.resample(X[(y == cl).flatten()], n_samples=mn)
        y_i = np.array([cl] * len(x_i))
        balanced_target.append(y_i)
        balanced_data.append(x_i)

    balanced_data = np.concatenate(balanced_data, axis=0)
    balanced_target = np.concatenate(balanced_target, axis=0)
    return balanced_data, balanced_target


def add_train_data(self, data3d, segmentation, voxelsize_mm, nth=50):
    data3dr = imma.image_manipulation.resize_to_mm(data3d, voxelsize_mm, self.working_voxelsize_mm)
    segmentationr = imma.image_manipulation.resize_to_shape(segmentation, data3dr.shape)
    _add_to_training_data(self, data3dr, segmentationr, nth)
    # self._add_to_training_data(data3dr, segmentationr, nth)


def _fv(self, data3dr):
    return self.feature_function(data3dr, self.working_voxelsize_mm)


def _add_to_training_data(self, data3dr, segmentationr, nth=50):
    fv = self._fv(data3dr)
    data = fv[::nth]
    target = np.reshape(segmentationr, [-1, 1])[::nth]
    #         print "shape ", data.shape, "  ", target.shape

    if self.data is None:
        self.data = data
        self.target = target
    else:
        self.data = np.concatenate([self.data, data], 0)
        self.target = np.concatenate([self.target, target], 0)
        # self.cl.fit(data, target)
