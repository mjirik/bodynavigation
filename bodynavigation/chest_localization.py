#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import argparse

#import featurevector

#import apdb
#  apdb.set_trace();\
#import scipy.io
import numpy as np
import scipy
import scipy.ndimage
import skimage.measure

from imtools import misc, qmisc # https://github.com/mjirik/imtools

#sys.path.append("/home/mjirik/projects/pysegbase")
# sys.path.append("/home/mjirik/projects/lisa")


import matplotlib.pyplot as plt
import os.path as op

import csv
import os
import pickle

import copy
import glob
import io3d
# import pandas as pd
import scipy
import scipy.signal
# import sklearn
# import sklearn.naive_bayes
# import sklearn.tree
# import sklearn.mixture

from skimage.filters import threshold_otsu as otsu

""""""
try:
    import bodynavigation.body_navigation
    reload(bodynavigation.body_navigation)
    #"Importovano body_navigation"

except:
    #"Nelze iportovat body_navigation"
    try:
        import body_navigation
        reload(body_navigation)
    except:
        pass


# import imtools
import sed3
from io3d.misc import resize_to_shape, resize_to_mm
#from lisa import volumetry_evaluation
#import tiled_liver_statistics as lst
# from scipy.ndimage.filters import sobel as sobel
# from scipy.ndimage.filters import laplace as laplace

from skimage.morphology import convex_hull_image, convex_hull_object, label, closing, opening
from skimage import morphology
# import skimage


def read_data(orig_fname, ref_fname):
    """ Pomoci io3d nacte 3D obrazek """
    
    data3d_orig, metadata = io3d.datareader.read(orig_fname)
    vs_mm1 = metadata['voxelsize_mm']
    data3d_seg, metadata = io3d.datareader.read(ref_fname)
    vs_mm = metadata['voxelsize_mm']
    
    return data3d_orig, data3d_seg, vs_mm, vs_mm1

def make_data(pattern):
    """ Vrati CT data jednoho obrazku (pattern), a to jak orig, tak i seg """
    
    reload(bodynavigation.body_navigation)
    reload(sed3)
    
    sliver_reference_dir = op.expanduser(u"~\Documents\Py\lisatest\data\medical\orig\sliver07/training/")
    
    # ktere soubory chci vybrat
    orig_fnames = glob.glob(sliver_reference_dir + "*orig*" + pattern + ".mhd")
    ref_fnames = glob.glob(sliver_reference_dir + "*seg*" + pattern + ".mhd")
    
    print(orig_fnames)
    print(ref_fnames)
    
    orig_fnames.sort()
    ref_fnames.sort()
    
    orig_data, ref_data, vs_mm, vs_mm1 = read_data(orig_fnames[0], ref_fnames[0])
 
    return orig_data, ref_data, vs_mm, vs_mm1


def make_basic_improvement(orig_data, vs_mm, vs_mm_tmp=[1.5, 1.5, 1.5]):
    
    """ Prevzorkuje vstupni obrazek ... voxel size mm """
    data3dr_tmp = resize_to_mm(orig_data, vs_mm, vs_mm_tmp)              # resizovana orig_data na milimetry
    ss = bodynavigation.body_navigation.BodyNavigation(data3dr_tmp, vs_mm_tmp) # objet bodynavigation, vstupuje tam obrazek a voxelsize
    
    return ss, data3dr_tmp


def analyze_object(var):
    print("")
    print(type(var))
    print("")


def print_2D_gray(img):
    """ Vykresli 2D obrazek (rez) """
    
    plt.figure()
    plt.imshow(img, cmap = 'gray')
    plt.show()
    
    
def print_it_all(ss, data3dr_tmp, seg, pattern):
    """ Vykresli jednotlive rezy v danem smeru (axis) 
    do sed3.show_slices() vstupuji: 
        - obrazek pro danou instanci Bona (surovy obrazek), ktery
          zpracovavame
        - obrazek seg - vysledek segmentace, tedy vystup metody get_ribs(lungs, ...)) 
        - dalsi parametry jako jsou krok po rezech, osa, atd."""
        
    fig = plt.figure(figsize = (25,25))
    sed3.show_slices(
        data3dr_tmp,                                 # vychozi obrazek
        seg,                                         # vystup metody get_ribs(lungs, ...), tedy vysledek segmentace
        slice_step=10, 
        axis=0,
        flipV=True,
        flipH=False
    )


def print_it_all_front(ss, data3dr_tmp, seg, pattern):
    """ Vykresli jednotlive rezy v danem smeru (axis) 
    do sed3.show_slices() vstupuji: 
        - obrazek pro danou instanci Bona (surovy obrazek), ktery
          zpracovavame
        - obrazek seg - vysledek segmentace, tedy vystup metody get_ribs(lungs, ...)) 
        - dalsi parametry jako jsou krok po rezech, osa, atd."""
        
    fig = plt.figure(figsize = (25,25))
    sed3.show_slices(
        data3dr_tmp,                                 # vychozi obrazek
        seg.astype(np.int8),
        slice_step=10, 
        axis=1,
        flipV=True,
        flipH=False
    )


class ChestLocalization:
    
    def __init__(self, bona_object=None, data3dr_tmp=None):
        self.ss = bona_object
        self.data3dr_tmp = data3dr_tmp
        

    def print_it_all(self, ss, data3dr_tmp, seg, pattern):
        """ Vykresli jednotlive rezy v danem smeru (axis) 
        do sed3.show_slices() vstupuji: 
            - obrazek pro danou instanci Bona (surovy obrazek), ktery
              zpracovavame
            - obrazek seg - vysledek segmentace, tedy vystup metody get_ribs(lungs, ...)) 
            - dalsi parametry jako jsou krok po rezech, osa, atd."""
            
        fig = plt.figure(figsize = (25,25))
        sed3.show_slices(
            data3dr_tmp,                                 # vychozi obrazek
            seg,                                         # vystup metody get_ribs(lungs, ...), tedy vysledek segmentace
            slice_step=10, 
            axis=0,
            flipV=True,
            flipH=False
        )
        #fig.savefig("output/"+str(pattern)+".png")
        
    def print_it_all_front(self, ss, data3dr_tmp, seg, pattern):
        """ Vykresli jednotlive rezy v danem smeru (axis) 
        do sed3.show_slices() vstupuji: 
            - obrazek pro danou instanci Bona (surovy obrazek), ktery
              zpracovavame
            - obrazek seg - vysledek segmentace, tedy vystup metody get_ribs(lungs, ...)) 
            - dalsi parametry jako jsou krok po rezech, osa, atd."""
            
        fig = plt.figure(figsize = (25,25))
        sed3.show_slices(
            data3dr_tmp,                                 # vychozi obrazek
            seg.astype(np.int8),
            slice_step=10, 
            axis=1,
            flipV=True,
            flipH=False
        )
        #fig.savefig("output/front"+str(pattern)+".png")


    def save_obj(self, obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    
    def load_obj(self, name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    
    
    def strict_intensity_filter(self, data3dr_tmp, strict_floor=300, min_floor=220):
        """ Vrati filtr, kde vezme vsechny objekty s velmi vysokou intenzitou a jejich okoli 3 pixely """
        
        #blur = scipy.ndimage.filters.gaussian_filter(copy.copy(data3dr_tmp), sigma=[1, 2, 2])
        hrube = data3dr_tmp > strict_floor
        #print_it_all(ss, data3dr_tmp, hrube*2, "001")
    
        dh = scipy.ndimage.morphology.distance_transform_edt(1-hrube)
    
        int_fil = (dh<5) & (data3dr_tmp>min_floor)
    
        #print_it_all(ss, data3dr_tmp, int_fil, "001")
        
        return int_fil
    
    def deep_struct_filter(self, data3dr_tmp, body):
        """ Najde vsechny hluboke (v ose z) objekty a jejich okoli 5 pixelu """
        
        min_bone_thr = 220
        bone = (data3dr_tmp>min_bone_thr) & body
        bone_sum = np.sum(bone, axis = 0) #> 150
    
        bone_mask = bone_sum>=1
        bone_hull = convex_hull_image(bone_mask)#convex_hull_image(bone_mask)
    
        def weak_dist_bone(self, bone_hull, body):
            """ Metoda pro zjisteni, jak blizko jsme stredu (nejvzdalenejsimu mistu od povrchu) """
            blank_body = np.ones(body.shape)
            bone_edge_dist = scipy.ndimage.morphology.distance_transform_edt(bone_hull)
            bone_edge_dist_maximum = np.max(bone_edge_dist)
            bone_edge_dist_focus = bone_edge_dist > 0*bone_edge_dist_maximum
            blank_body[:] = bone_edge_dist_focus
            ld = scipy.ndimage.morphology.distance_transform_edt(blank_body)
            return resize_to_shape(ld, self.ss.orig_shape)
    
        dh = weak_dist_bone(bone_hull, body)
    
        blur = scipy.ndimage.filters.gaussian_filter(copy.copy(data3dr_tmp), sigma=[17, 3, 3]) > 100
        db = scipy.ndimage.morphology.distance_transform_edt(1-blur)
        dbf = db >= 6
    
        struct_filter = ((dbf==False) & (dh>15))==False
        return struct_filter
    
    def deep_struct_filter_old(self, data3dr_tmp):
        blur = scipy.ndimage.filters.gaussian_filter(copy.copy(data3dr_tmp), sigma=[17, 3, 3]) > 150
        db = scipy.ndimage.morphology.distance_transform_edt(1-blur)
        dbf = db>4
    
        struct_filter = dbf
        return struct_filter
    
    
    def internal_filter(self, lungs, coronal, body):
        #lungs = ss.get_lungs()
        lungs_sum = np.sum(lungs, axis = 0)>=5
        lungs_hull = convex_hull_image(lungs_sum)
    
        def dist_lungs(lungs_hull, body):
            """ Metoda pro zjisteni, jak blizko jsme stredu (nejvzdalenejsimu mistu od povrchu) """
            blank_body = np.ones(body.shape)
            lungs_edge_dist = scipy.ndimage.morphology.distance_transform_edt(lungs_hull)
            lungs_edge_dist_maximum = np.max(lungs_edge_dist)
            lungs_edge_dist_focus = lungs_edge_dist > 0.2 * lungs_edge_dist_maximum               # 0.1 puvodne
            blank_body[:] = lungs_edge_dist_focus
            ld = scipy.ndimage.morphology.distance_transform_edt(blank_body)
            # resized_to_orig = resize_to_shape(ld, self.ss.orig_shape)
            # return resized_to_orig
            return ld

        dist_lungs_with_body = dist_lungs(lungs_hull, body)
        lungs_mask = (dist_lungs_with_body > 0 ) & (coronal > 0)
        #print_2D_gray(lungs_mask[100])
    
        #print_it_all(ss, data3dr_tmp, lungs_mask*3, "001")
    
    
        def improve_lungs(dist_lungs_hulls, final_area_filter):
            """ Upravi masku lungs pro specialni ucely """
            lungs_hulls = dist_lungs_hulls > 0
            new_filter = np.zeros(final_area_filter.shape)
            for i, area_slice in enumerate(final_area_filter):
                convex = convex_hull_image(area_slice)
                new_filter[i] = lungs_hulls[i] & convex
            return new_filter
        
        return lungs_mask
    
    
    def clear_body(self, body):
        """ Vycisti obraz od stolu a ostatnich veci okolo tela a zanecha pouze a jen telo """
        
        body = scipy.ndimage.filters.gaussian_filter(copy.copy(body).astype(float), sigma=[15, 0, 0]) > 0.7
        body = morphology.remove_small_objects(body, 2400)
        
        body[0] = False
        body[-1] = False
        
        body = scipy.ndimage.filters.gaussian_filter(copy.copy(body.astype(float)), sigma=[1, 3, 3]) > 0.2
    
        bodylabel = label(body)
    
        n_of_pixels = [np.count_nonzero(bodylabel==i) for i in range(len(np.unique(bodylabel)))]
        labelsort = np.argsort(n_of_pixels)[::-1]
    
        newbody = np.zeros(body.shape)
        newbody[bodylabel==labelsort[0]] = body[(bodylabel==labelsort[0])]
        newbody[bodylabel==labelsort[1]] = body[(bodylabel==labelsort[1])]
    
        return newbody.astype(bool)
    
    def area_filter(self, data3dr_tmp, body, lungs, coronal):
        min_bone_thr = 220
        
        #body = clear_body(body)
        #blured = scipy.ndimage.filters.gaussian_filter(copy.copy(data3dr_tmp), sigma=[1, 3, 3]) >= 120
          
        def get_bone_hull(data3dr_tmp, body):
            # jeste by bylo dobre u patere vynechat konvexni obal
            bone = (data3dr_tmp>min_bone_thr) & body
            mask = np.zeros(data3dr_tmp.shape)
            
            step = 32
            for i in range(len(data3dr_tmp)):
                mask[i] = np.sum(
                    bone[
                    int(max(0, i - step / 2)):
                    int(min(len(data3dr_tmp), i + step / 2))
                    ], axis = 0
                ) >= 1
                try:
                    mask[i] = convex_hull_image(mask[i])
                except:
                    pass
            
            mask = mask>0
            return mask
        
        bone_step_hull = get_bone_hull(data3dr_tmp, body)
    
        bone = (data3dr_tmp>min_bone_thr) & body
        bone_sum = np.sum(bone, axis = 0) #> 150
        #bone_sum = scipy.ndimage.filters.gaussian_filter(copy.copy(bone_sum), sigma=[3, 3])
    
        bone_mask = bone_sum>=1
        bone_mask = scipy.ndimage.filters.gaussian_filter(copy.copy(bone_mask.astype(float)), sigma=[3, 3])>0.2
        bone_hull = scipy.ndimage.morphology.binary_fill_holes(bone_mask)#convex_hull_image(bone_mask)
    
        def dist_bone(bone_hull, body):
            """ Metoda pro zjisteni, jak blizko jsme stredu (nejvzdalenejsimu mistu od povrchu) """
            blank_body = np.ones(body.shape)
            bone_edge_dist = scipy.ndimage.morphology.distance_transform_edt(bone_hull)
            bone_edge_dist_maximum = np.max(bone_edge_dist)
            bone_edge_dist_focus = bone_edge_dist > 0*bone_edge_dist_maximum
            blank_body[:] = bone_edge_dist_focus
            ld = scipy.ndimage.morphology.distance_transform_edt(blank_body)
            # ld_resized = resize_to_shape(ld, self.ss.orig_shape)
            # return ld_resized
            return ld
        
        def improve_bone_hull(bone_area_filter):
            basic_loc_filter = body & (coronal > 0)
            mask = np.zeros(bone_area_filter.shape)
            
            for i in range(len(bone_area_filter)):
                try:
                    down_mask = bone_area_filter[i] & basic_loc_filter[i]
                    mask[i] = convex_hull_image(down_mask) | bone_area_filter[i]
                    mask[i] = scipy.ndimage.morphology.binary_fill_holes(mask[i])
                except:
                    pass
            mask = mask > 0
            return mask
        
        bone_hulls = dist_bone(bone_hull, body)
        bone_area_filter = improve_bone_hull(bone_hulls > 0) # &(bone_hulls<7)

        # v internal_filter nebyl žádný return
        lungs_internal_filter = self.internal_filter(lungs, coronal, body)
        
        final_area_filter = bone_area_filter & (bone_step_hull | lungs_internal_filter) & body
    
        return final_area_filter
    
    
    def area_mask(self, img, final_area_filter, body):
        
        spine = self.ss.get_spine()
        
        intensity_filter = (img > 150) & (spine==0) & body #& lungs_area_filter
        area_inv = final_area_filter
        dist_hull = scipy.ndimage.morphology.distance_transform_edt(area_inv)
           
        self.print_it_all(self.ss, area_inv, intensity_filter * 2, "001")
        
        to_save = area_inv.astype(float)+(img > 200).astype(float)
        
        return to_save
    
    
    def dist_hull(self, hull):
        d = scipy.ndimage.morphology.distance_transform_edt(hull)
        result = (self.data3dr_tmp>180) & (d<=8)
        return result
    
    
    def process_z_axe(self, ribs, lungs, pattern):
        pixels = list()
        for layer in ribs:
            pixels.append(np.sum(layer.astype(float)))
    
        sig = ( len(pixels) // 30)*2 + 1
    #    print "sigma = ",sig
        pixels_blured = scipy.ndimage.filters.gaussian_filter(pixels, sigma=sig) 
        
    #    print "lungs_center: ",np.max(np.where(lungs),axis = 1),
        lungs_zet_center = np.median(np.where(lungs), axis = 1).astype(int)[0]
    #    print "lungs_zet_center: ",lungs_zet_center
        
        extrem = scipy.signal.argrelextrema(pixels_blured, np.less)[0]
        
        border = 0
        extrem = [e for e in extrem if e < lungs_zet_center]
        if len(extrem) >= 1:
            border = np.max(extrem)
    #        print "extrem: ",extrem, type(extrem), "mez: ", border
    #    else:
    #        print "extrem: zadny nenalezen => mez: ", 0
    #    
    #    print "______________________________________________"
    #    
    #    plt.figure()
    #    plt.plot(pixels)
    #    plt.plot(pixels_blured) 
    #    plt.plot([border, border], [0, np.max(pixels)])
    #    plt.grid()
    #    plt.show()
    #    plt.savefig("z_osa_"+pattern+".png")
        
        return border
    
        
        
