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
from PIL import Image, ImageDraw
import skimage.transform

sys.path.append("..")
import bodynavigation.organ_detection
print("bodynavigation.organ_detection path:", os.path.abspath(bodynavigation.organ_detection.__file__))
from bodynavigation.organ_detection import OrganDetection

import io3d

def drawPoints(draw, points, colour=(255,0,0), outline=None, size=5):
    for p in points: # p = [x,y]
        xy = [p[0], p[1], p[0]+size, p[1]+size]
        draw.rectangle(xy, fill=colour, outline=outline)

def drawPointsTo3DData(data3d, voxelsize, point_sets = []):
    """
    point_sets = [[points, colour=(255,0,0), outline=None],...]
    Returns RGB Image object
    """

    data3d[ data3d < -1024 ] = -1024
    data3d[ data3d > 1024 ] = 1024
    data3d = data3d + abs(np.min(data3d))

    view_z = np.sum(data3d, axis=0, dtype=np.int32).astype(np.float)
    view_z = (view_z*(255.0/view_z.max())).astype(np.int32)

    view_y = np.sum(data3d, axis=1, dtype=np.int32).astype(np.float)
    view_y = (view_y*(255.0/view_y.max())).astype(np.int32)

    view_x = np.sum(data3d, axis=2, dtype=np.int32).astype(np.float)
    view_x = (view_x*(255.0/view_x.max())).astype(np.int32)

    new_shape = (int(data3d.shape[1] * voxelsize[1]), int(data3d.shape[2] * voxelsize[2]))
    view_z = skimage.transform.resize(
            view_z, new_shape, order=3, mode="reflect", clip=True, preserve_range=True,
            ).astype(np.int32)

    new_shape = (int(data3d.shape[0] * voxelsize[0]), int(data3d.shape[1] * voxelsize[1]))
    view_y = skimage.transform.resize(
            view_y, new_shape, order=3, mode="reflect", clip=True, preserve_range=True,
            ).astype(np.int32)

    new_shape = (int(data3d.shape[0] * voxelsize[0]), int(data3d.shape[2] * voxelsize[2]))
    view_x = skimage.transform.resize(
            view_x, new_shape, order=3, mode="reflect", clip=True, preserve_range=True,
            ).astype(np.int32)

    # draw view_z
    img = Image.fromarray(view_z, 'I').convert("RGB")
    draw = ImageDraw.Draw(img)
    for pset in point_sets:
        points, colour, outline = tuple(pset)
        points = [ list(np.asarray(p)*voxelsize) for p in points ]
        z, y, x = zip(*points)
        points_2d = zip(x, y)
        drawPoints(draw, points_2d, colour=colour, outline=outline)
    img_z = img; del(draw)

    # draw view_y
    img = Image.fromarray(view_y, 'I').convert("RGB")
    draw = ImageDraw.Draw(img)
    for pset in point_sets:
        points, colour, outline = tuple(pset)
        points = [ list(np.asarray(p)*voxelsize) for p in points ]
        z, y, x = zip(*points)
        points_2d = zip(x, z)
        drawPoints(draw, points_2d, colour=colour, outline=outline)
    img_y = img; del(draw)

    # draw view_x
    img = Image.fromarray(view_x, 'I').convert("RGB")
    draw = ImageDraw.Draw(img)
    for pset in point_sets:
        points, colour, outline = tuple(pset)
        points = [ list(np.asarray(p)*voxelsize) for p in points ]
        z, y, x = zip(*points)
        points_2d = zip(y, z)
        drawPoints(draw, points_2d, colour=colour, outline=outline)
    img_x = img; del(draw)

    # connect and retorn images
    img = Image.new('RGB', (max(img_y.size[0]+img_x.size[0], img_z.size[0]), \
        max(img_y.size[1]+img_z.size[1], img_x.size[1]+img_z.size[1])))

    img.paste(img_y, (0,0))
    img.paste(img_x, (img_y.size[0],0))
    img.paste(img_z, (0,max(img_y.size[1], img_x.size[1])))
    #img.show(); sys.exit(0)

    return img


def main():
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # input parser
    parser = argparse.ArgumentParser(description="Batch Processing")
    parser.add_argument('-i','--datadirs', default=None,
            help='path to dir with data dirs')
    parser.add_argument('-o','--outputdir', default="./batch_output",
            help='path to output dir')
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

    for dirname in sorted(next(os.walk(args.datadirs))[1]):
        datapath = os.path.abspath(os.path.join(args.datadirs, dirname))
        print("Processing: ", datapath)

        try:
            data3d, metadata = io3d.datareader.read(datapath)
            voxelsize = metadata["voxelsize_mm"]
            obj = OrganDetection(data3d, voxelsize)

            points_spine, points_hip_joint = obj.analyzeBones() # in voxels

            del(obj)

            img = drawPointsTo3DData(data3d, voxelsize, point_sets = [ \
                [points_spine, (255,0,0), None],
                [points_hip_joint, (0,255,0), (0,0,0)]
                ])

            img.save(os.path.join(outputdir, "%s.png" % dirname))
            #img.show()

        except:
            print("EXCEPTION! SAVING TRACEBACK!")
            with open(os.path.join(outputdir, "%s.txt" % dirname), 'w') as f:
                f.write(traceback.format_exc())

if __name__ == "__main__":
    main()
