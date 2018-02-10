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

            data3d[ data3d < -1024 ] = -1024
            data3d[ data3d > 1024 ] = 1024
            data3d = data3d + abs(np.min(data3d))
            out = np.sum(data3d, axis=1, dtype=np.int32).astype(np.float)
            out *= (255.0/out.max())
            out = out.astype(np.int32)

            new_shape = (int(out.shape[0] * voxelsize[0]), int(out.shape[1] * voxelsize[2]))
            out = skimage.transform.resize(
                    out, new_shape, order=3, mode="reflect", clip=True, preserve_range=True,
                    ).astype(np.int32)

            img = Image.fromarray(out, 'I').convert("RGB")
            draw = ImageDraw.Draw(img)
            for p in points_spine:
                p = np.asarray(p)*voxelsize
                xy = [p[2], p[0], min(out.shape[1],p[2]+5), min(out.shape[0],p[0]+5)]
                draw.rectangle(xy, fill=(255,0,0))
            for p in points_hip_joint:
                p = np.asarray(p)*voxelsize
                xy = [p[2], p[0], min(out.shape[1],p[2]+5), min(out.shape[0],p[0]+5)]
                draw.rectangle(xy, fill=(0,255,0), outline=(0,0,0))
            del(draw)

            img.save(os.path.join(outputdir, "%s.png" % dirname))
            #img.show()

        except:
            print("EXCEPTION! SAVING TRACEBACK!")
            with open(os.path.join(outputdir, "%s.txt" % dirname), 'w') as f:
                f.write(traceback.format_exc())

if __name__ == "__main__":
    main()
