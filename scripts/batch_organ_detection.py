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

from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw
import skimage.transform

sys.path.append("..")
import bodynavigation.organ_detection
print("bodynavigation.organ_detection path:", os.path.abspath(bodynavigation.organ_detection.__file__))
from bodynavigation.organ_detection import OrganDetection

import io3d
import sed3

def drawPoints(img, points, axis, colour=(255,0,0,255), outline=None, size=1):
    if len(points) == 0: return img
    if len(colour)==3:
        colour = (colour[0],colour[1],colour[2],255)

    z, y, x = zip(*points)
    if axis == 0:
        points_2d = zip(x, y)
    elif axis == 1:
        points_2d = zip(x, z)
    elif axis == 2:
        points_2d = zip(y, z)
    else:
        raise Exception("Invalid axis value: %s" % str(axis))

    img_d = Image.new('RGBA', img.size)

    draw = ImageDraw.Draw(img_d)
    for p in points_2d: # p = [x,y]

        xy = [p[0]-(size/2), p[1]-(size/2), p[0]+(size/2), p[1]+(size/2)]
        draw.rectangle(xy, fill=colour, outline=outline)
    del(draw)

    img = Image.composite(img_d, img, img_d)
    return img

def drawVolume(img, mask, colour=(255,0,0,100)):
    if len(colour)==3:
        colour = (colour[0],colour[1],colour[2],255)
    img_mask = np.zeros((mask.shape[0],mask.shape[1], 4), dtype=np.uint8)
    img_mask[:,:,0] = colour[0]
    img_mask[:,:,1] = colour[1]
    img_mask[:,:,2] = colour[2]
    img_mask[:,:,3] = mask.astype(np.uint8)*colour[3]
    img_mask = Image.fromarray(img_mask, 'RGBA')
    #img_mask.show()
    img = Image.composite(img_mask, img, img_mask)
    return img

def drawPointsTo3DData(data3d, voxelsize, point_sets = [], volume_sets = []):
    """
    point_sets = [[points, colour=(255,0,0), outline=None, size=3],...]
    volume_sets = [[mask, colour=(255,0,0)],...]
    Returns RGB Image object
    """

    data3d[ data3d < -1024 ] = -1024
    data3d[ data3d > 1024 ] = 1024
    data3d = data3d + abs(np.min(data3d))

    # axis views
    view_z = np.sum(data3d, axis=0, dtype=np.int32).astype(np.float)
    view_z = (view_z*(255.0/view_z.max())).astype(np.int32)

    view_y = np.sum(data3d, axis=1, dtype=np.int32).astype(np.float)
    view_y = (view_y*(255.0/view_y.max())).astype(np.int32)

    view_x = np.sum(data3d, axis=2, dtype=np.int32).astype(np.float)
    view_x = (view_x*(255.0/view_x.max())).astype(np.int32)

    tmp = []
    for vset in volume_sets:
        mask, colour = tuple(vset)
        mask_z = np.sum(mask.astype(np.uint32), axis=0, dtype=np.uint32) != 0
        mask_y = np.sum(mask.astype(np.uint32), axis=1, dtype=np.uint32) != 0
        mask_x = np.sum(mask.astype(np.uint32), axis=2, dtype=np.uint32) != 0
        tmp.append(([mask_z, mask_y, mask_x], colour))
    volume_sets = tmp

    # resize to 1x1x1 voxelsize
    new_shape_z = (int(data3d.shape[1] * voxelsize[1]), int(data3d.shape[2] * voxelsize[2]))
    new_shape_y = (int(data3d.shape[0] * voxelsize[0]), int(data3d.shape[1] * voxelsize[1]))
    new_shape_x = (int(data3d.shape[0] * voxelsize[0]), int(data3d.shape[2] * voxelsize[2]))

    view_z = skimage.transform.resize(
            view_z, new_shape_z, order=3, mode="reflect", clip=True, preserve_range=True,
            ).astype(np.int32)
    view_y = skimage.transform.resize(
            view_y, new_shape_y, order=3, mode="reflect", clip=True, preserve_range=True,
            ).astype(np.int32)
    view_x = skimage.transform.resize(
            view_x, new_shape_x, order=3, mode="reflect", clip=True, preserve_range=True,
            ).astype(np.int32)

    tmp = []
    for pset in point_sets:
        points, colour, outline, size = tuple(pset)
        points = [ list(np.asarray(p)*voxelsize) for p in points ]
        tmp.append((points, colour, outline, size))
    point_sets = tmp

    tmp = []
    for vset in volume_sets:
        masks, colour = tuple(vset)
        mask_z, mask_y, mask_x = tuple(masks)
        mask_z = skimage.transform.resize(
            mask_z, new_shape_z, order=0, mode="reflect", clip=True, preserve_range=True,
            ).astype(np.bool)
        #ed = sed3.sed3(np.expand_dims(mask_z.astype(np.int8), axis=0)); ed.show()
        mask_y = skimage.transform.resize(
            mask_y, new_shape_y, order=0, mode="reflect", clip=True, preserve_range=True,
            ).astype(np.bool)
        #ed = sed3.sed3(np.expand_dims(mask_y.astype(np.int8), axis=0)); ed.show()
        mask_x = skimage.transform.resize(
            mask_x, new_shape_x, order=0, mode="reflect", clip=True, preserve_range=True,
            ).astype(np.bool)
        #ed = sed3.sed3(np.expand_dims(mask_x.astype(np.int8), axis=0)); ed.show()
        tmp.append(([mask_z, mask_y, mask_x], colour))
    volume_sets = tmp

    # draw view_z
    img = Image.fromarray(view_z, 'I').convert("RGBA")
    for vset in volume_sets:
        masks, colour = tuple(vset)
        mask_z, mask_y, mask_x = tuple(masks)
        img = drawVolume(img, mask_z, colour)
    for pset in point_sets:
        points, colour, outline, size = tuple(pset)
        img = drawPoints(img, points, axis=0, colour=colour, outline=outline, size=size)
    img_z = img

    # draw view_y
    img = Image.fromarray(view_y, 'I').convert("RGBA")
    for vset in volume_sets:
        masks, colour = tuple(vset)
        mask_z, mask_y, mask_x = tuple(masks)
        img = drawVolume(img, mask_y, colour)
    for pset in point_sets:
        points, colour, outline, size = tuple(pset)
        img = drawPoints(img, points, axis=1, colour=colour, outline=outline, size=size)
    img_y = img

    # draw view_x
    img = Image.fromarray(view_x, 'I').convert("RGBA")
    for vset in volume_sets:
        masks, colour = tuple(vset)
        mask_z, mask_y, mask_x = tuple(masks)
        img = drawVolume(img, mask_x, colour)
    for pset in point_sets:
        points, colour, outline, size = tuple(pset)
        img = drawPoints(img, points, axis=2, colour=colour, outline=outline, size=size)
    img_x = img

    # connect and retorn images
    img = Image.new('RGBA', (max(img_y.size[0]+img_x.size[0], img_z.size[0]), \
        max(img_y.size[1]+img_z.size[1], img_x.size[1]+img_z.size[1])))

    img.paste(img_y, (0,0))
    img.paste(img_x, (img_y.size[0],0))
    img.paste(img_z, (0,max(img_y.size[1], img_x.size[1])))
    #img.show(); sys.exit(0)

    return img.convert("RGB")

def interpolatePointsZ(points, step=0.1):
    if len(points) <= 1: return points

    z, y, x = zip(*points)
    z_new = list(np.arange(z[0], z[-1]+1, step))

    zz1 = np.polyfit(z, y, 3)
    f1 = np.poly1d(zz1)
    y_new = f1(z_new)

    zz2 = np.polyfit(z, x, 3)
    f2 = np.poly1d(zz2)
    x_new = f2(z_new)

    points = [ tuple([z_new[i], y_new[i], x_new[i]]) for i in range(len(z_new)) ]
    return points

def processData(datapath, name, outputdir):
    try:
        print("Processing: ", datapath)

        data3d, metadata = io3d.datareader.read(datapath)
        voxelsize = metadata["voxelsize_mm"]
        obj = OrganDetection(data3d, voxelsize)

        bones_stats = obj.analyzeBones() # in voxels
        # body = obj.getBody()
        # fatlessbody = obj.getFatlessBody()
        # bones = obj.getBones()
        # lungs = obj.getLungs()
        vessels = obj.getVessels()
        vessels_stats = obj.analyzeVessels() # in voxels

        # ed = sed3.sed3(body); ed.show()
        # ed = sed3.sed3(fatlessbody); ed.show()
        # ed = sed3.sed3(bones); ed.show()
        # ed = sed3.sed3(lungs); ed.show()
        # ed = sed3.sed3(vessels); ed.show()

        del(obj)

        img = drawPointsTo3DData(data3d, voxelsize, point_sets = [ \
                [interpolatePointsZ(bones_stats["spine"], step=0.1), (255,0,0), None, 1],
                [bones_stats["hip_joints"], (0,255,0), (0,0,0), 7],
                [interpolatePointsZ(vessels_stats["aorta"], step=0.1), (0,255,255), None, 1],
                [interpolatePointsZ(vessels_stats["vena_cava"], step=0.1), (255,0,255), None, 1]
            ], volume_sets = [ \
                [vessels, (0,0,255,100)]
            ])

        img.save(os.path.join(outputdir, "%s.png" % name))
        #img.show()

    except:
        print("EXCEPTION! SAVING TRACEBACK!")
        with open(os.path.join(outputdir, "%s.txt" % name), 'w') as f:
            f.write(traceback.format_exc())

def processThread(args):
    processData(*args)

def main():
    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # input parser
    parser = argparse.ArgumentParser(description="Batch Processing. Needs to be SIGKILLed to terminate")
    parser.add_argument('-i','--datadirs', default=None,
            help='path to dir with data dirs')
    parser.add_argument('-o','--outputdir', default="./batch_output",
            help='path to output dir')
    parser.add_argument('-t','--threads', type=int, default=1,
            help='How many processes (CPU cores) to use. Max MEM usage for smaller data is around 2.5GB, big ones can go over 8GB.')
    parser.add_argument("-d", "--debug", action="store_true",
            help='run in debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.datadirs is None:
        logger.error("Missing data directory path --datadirs")
        sys.exit(2)
    elif not os.path.exists(args.datadirs) or os.path.isfile(args.datadirs):
        logger.error("Invalid data directory path --datadirs")
        sys.exit(2)

    outputdir = os.path.abspath(args.outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    inputs = []
    for dirname in sorted(next(os.walk(args.datadirs))[1]):
        datapath = os.path.abspath(os.path.join(args.datadirs, dirname))
        inputs.append([datapath, dirname, outputdir])

    pool = Pool(processes=args.threads)
    pool.map(processThread, inputs)

if __name__ == "__main__":
    main()
