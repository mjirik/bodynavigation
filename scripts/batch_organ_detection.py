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
import resource

import numpy as np

sys.path.append("..")
import bodynavigation.organ_detection
print("bodynavigation.organ_detection path:", os.path.abspath(bodynavigation.organ_detection.__file__))
from bodynavigation.organ_detection import OrganDetection
from bodynavigation.results_drawer import ResultsDrawer

import io3d

"""
/usr/bin/time -v python batch_organ_detection.py -d -o ./batch_output/ -i ../test_data_3Dircadb1/ --dump ../READY_DIR_NEW/ -p "lungs,bones,bones_stats,kidneys" -t 3
"""

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

def processData(datapath, name, outputdir, parts=[], dumpdir=None, readypath=None, memorylimit=-1, draw_depth=False):
    try:
        print("Processing: ", datapath)

        # set memory limit
        if memorylimit > 0: # resource.RLIMIT_AS works with virtual memory
            print("Setting memory limit to: %s GB" % str(memorylimit))
            memorylimit_b = int(memorylimit*(2**30))
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (memorylimit_b, min(memorylimit_b,hard)))

        onlyfiles = sorted([f for f in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, f))])
        if len(onlyfiles) == 1:
            datapath = os.path.join(datapath, onlyfiles[0])
            print("Only one file datapath! Changing datapath to: ", datapath)
        else:
            for f in onlyfiles:
                if f.strip().lower().endswith(".mhd"):
                    datapath = os.path.join(datapath, f)
                    print("Detected *.mhd file! Changing datapath to: ", datapath)
                    break

        data3d = None
        if readypath is None:
            data3d, metadata = io3d.datareader.read(datapath, dataplus_format=False)
            voxelsize = metadata["voxelsize_mm"]
            obj = OrganDetection(data3d, voxelsize)
        else:
            print("Loading preprocessed data from readypath: ", readypath)
            obj = OrganDetection.fromDirectory(os.path.abspath(readypath))
            voxelsize = obj.spacing_source

        # always use original data3d if availible
        if data3d is None:
            try:
                data3d, metadata = io3d.datareader.read(datapath, dataplus_format=False)
            except:
                print("Failed to load data3d from datapath! Using obj.getData3D().")
                data3d = obj.getData3D()

        point_sets = []; volume_sets = []
        if draw_depth:
            rd = ResultsDrawer(mask_depth = True, default_volume_alpha = 255)
        else:
            rd = ResultsDrawer(default_volume_alpha=100)

        parts_stats = [x for x in parts if x.endswith("_stats")]
        parts_masks = [x for x in parts if (x not in parts_stats)]

        COLOR_IDX = {
            "body":7, "fatlessbody":8, "lungs":9, "bones":2, "vessels":0,
            "kidneys":6, "liver":4, "spleen":7, "diaphragm":10,
            }
        # for k in COLOR_IDX: # TODO - remove
        #     COLOR_IDX[k] = COLOR_IDX["bones"]
        i = 0
        for p in parts:
            if p not in COLOR_IDX:
                while i in list(COLOR_IDX.values()): i+=1
                COLOR_IDX[p] = i

        for part in parts_masks:
            volume_sets.append([obj.getPart(part), {"color":rd.getRGBA(COLOR_IDX[part])}])

        for ps in parts_stats:
            part = ps.replace("_stats", "")
            stats = obj.analyzePart(part)

            if ps == "bones_stats":
                point_sets.append([interpolatePointsZ(stats["spine"], step=0.1), {"color":rd.getRGBA(2), "border":(0,0,0), "size":1}])
                point_sets.append([stats["hips_joints"], {"color":(0,255,0), "border":(0,0,0), "size":7}])
                point_sets.append([stats["hips_start"], {"color":(0,0,255), "border":(0,0,0), "size":7}])

            elif ps == "vessels_stats":
                point_sets.append([interpolatePointsZ(stats["aorta"], step=0.1), {"color":rd.getRGBA(6), "border":(0,0,0), "size":1}])
                point_sets.append([interpolatePointsZ(stats["vena_cava"], step=0.1), {"color":rd.getRGBA(7), "border":(0,0,0), "size":1}])
                point_sets.append([stats["liver"], {"color":rd.getRGBA(2), "border":(0,0,0), "size":7}])

        img = rd.drawImage(data3d, voxelsize, point_sets=point_sets, volume_sets=volume_sets)
        img.save(os.path.join(outputdir, "%s.png" % name))
        #img.show()

        if dumpdir is not None:
            if not os.path.exists(dumpdir): os.makedirs(dumpdir)
            obj.toDirectory(dumpdir)

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
            help="How many processes (CPU cores) to use. Max expected MEM usage for abdomen only data is around 2GB, for whole body data it's around 5GB.")
    parser.add_argument('-m','--memorylimit', type=int, default=-1,
            help='How many GB of VIRTUAL memory are individual threads allowed before they are terminated. Might only work on Unix systems. Default is unlimited')
    parser.add_argument('-p','--parts', default="bones_stats,vessels,vessels_stats",
            help='Body parts to process sparated by ",", Use "None" to disable, defaults: "bones_stats,vessels,vessels_stats"')
    parser.add_argument("--dump", default=None,
            help='dump all processed data to dir in path')
    parser.add_argument('-r','--readydirs', default=None,
            help='path to dir with dirs with preporcessed data3d.dcm and masks')
    parser.add_argument("--drawdepth", action="store_true",
            help='draw image in solid depth mode.')
    parser.add_argument("-d", "--debug", action="store_true",
            help='run in debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    logging.getLogger("io3d").setLevel(logging.WARNING)

    if args.datadirs is None:
        logger.error("Missing data directory path --datadirs")
        sys.exit(2)
    elif not os.path.exists(args.datadirs) or os.path.isfile(args.datadirs):
        logger.error("Invalid data directory path --datadirs")
        sys.exit(2)

    outputdir = os.path.abspath(args.outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    parts = []
    if args.parts.strip().lower() != "none":
        parts = [ s.strip().lower() for s in args.parts.split(",") ]

    ready_dirnames = []
    if args.readydirs is not None:
        ready_dirnames = sorted(next(os.walk(args.readydirs))[1])

    inputs = []
    for dirname in sorted(next(os.walk(args.datadirs))[1]):
        datapath = os.path.abspath(os.path.join(args.datadirs, dirname))
        dumpdir = None
        if args.dump is not None:
            dumpdir = os.path.join(os.path.abspath(args.dump), dirname)
        readypath = None
        if dirname in ready_dirnames:
            readypath = os.path.abspath(os.path.join(args.readydirs, dirname))
        inputs.append([datapath, dirname, outputdir, parts, dumpdir, readypath, args.memorylimit, args.drawdepth])

    pool = Pool(processes=args.threads)
    pool.map(processThread, inputs)
    pool.terminate() # pool.close()
    pool.join()

if __name__ == "__main__":
    main()
