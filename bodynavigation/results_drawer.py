#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enable Python3 code in Python2 - Must be first in file!
from __future__ import print_function   # print("text")
from __future__ import division         # 2/3 == 0.666; 2//3 == 0
from __future__ import absolute_import  # 'import submodule2' turns into 'from . import submodule2'
from builtins import range              # replaces range with xrange

import logging
logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image, ImageDraw
import skimage.transform


class ResultsDrawer(object):

    COLORS = [
        (255,0,0), (255,106,0), (255,213,0), (191,255,0), (0,255,21), \
        (0,255,234), (0,170,255), (43,0,255), (255,0,255), (255,0,149)
        ]

    def __init__(self, data3d_forced_min = -1024, data3d_forced_max = 1024,\
            default_volume_alpha = 100, default_point_alpha = 255, \
            default_point_border = (0,0,0), default_point_size = 5 ):
        self.data3d_forced_min = data3d_forced_min
        self.data3d_forced_max = data3d_forced_max
        self.default_volume_alpha = default_volume_alpha
        self.default_point_alpha = default_point_alpha
        if default_point_border is None:
            self.default_point_border = None
        else:
            self.default_point_border = tuple(list(default_point_border)+[default_point_alpha,])
        self.default_point_size = default_point_size

    def getRGBA(self, idx, a=255):
        """ Returns RGBA color with specified intex """
        while idx >= len(self.COLORS):
            idx = idx - len(self.COLORS)
        c = list(self.COLORS[idx]); c.append(a)
        return tuple(c)

    def _drawPoints(self, img, points, axis, colour=(255,0,0,255), outline=None, size=1):
        """
        img - 2D array
        points - 3D coordinates
        axis - on which axis is 'img' image of
        """
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

        # draw outline/border
        if outline is not None:
            bsize = size+2
            for p in points_2d: # p = [x,y]
                xy = [p[0]-(bsize/2), p[1]-(bsize/2), p[0]+(bsize/2), p[1]+(bsize/2)]
                draw.rectangle(xy, fill=outline)

        # draw points
        for p in points_2d: # p = [x,y]
            xy = [p[0]-(size/2), p[1]-(size/2), p[0]+(size/2), p[1]+(size/2)]
            draw.rectangle(xy, fill=colour)

        img = Image.composite(img_d, img, img_d)
        return img

    def _drawVolume(self, img, mask, colour=(255,0,0,100)):
        """
        img, mask - 2D arrays
        """
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

    def drawImage(self, data3d, voxelsize, point_sets = [], volume_sets = []):
        """
        point_sets = [[points, colour=(255,0,0,100), outline=None, size=3],...]
        volume_sets = [[mask, colour=(255,0,0,100)],...]
        Returns RGB Image object

        Save with: img.save(os.path.join(outputdir, "%s.png" % name))
        Open with: img.show()
        """

        data3d[ data3d < self.data3d_forced_min ] = self.data3d_forced_min
        data3d[ data3d > self.data3d_forced_max ] = self.data3d_forced_max
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
                view_z, new_shape_z, order=1, mode="reflect", clip=True, preserve_range=True,
                ).astype(np.int32)
        view_y = skimage.transform.resize(
                view_y, new_shape_y, order=1, mode="reflect", clip=True, preserve_range=True,
                ).astype(np.int32)
        view_x = skimage.transform.resize(
                view_x, new_shape_x, order=1, mode="reflect", clip=True, preserve_range=True,
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
            img = self._drawVolume(img, mask_z, colour)
        for pset in point_sets:
            points, colour, outline, size = tuple(pset)
            img = self._drawPoints(img, points, axis=0, colour=colour, outline=outline, size=size)
        img_z = img

        # draw view_y
        img = Image.fromarray(view_y, 'I').convert("RGBA")
        for vset in volume_sets:
            masks, colour = tuple(vset)
            mask_z, mask_y, mask_x = tuple(masks)
            img = self._drawVolume(img, mask_y, colour)
        for pset in point_sets:
            points, colour, outline, size = tuple(pset)
            img = self._drawPoints(img, points, axis=1, colour=colour, outline=outline, size=size)
        img_y = img

        # draw view_x
        img = Image.fromarray(view_x, 'I').convert("RGBA")
        for vset in volume_sets:
            masks, colour = tuple(vset)
            mask_z, mask_y, mask_x = tuple(masks)
            img = self._drawVolume(img, mask_x, colour)
        for pset in point_sets:
            points, colour, outline, size = tuple(pset)
            img = self._drawPoints(img, points, axis=2, colour=colour, outline=outline, size=size)
        img_x = img

        # connect images
        img = Image.new('RGBA', (max(img_y.size[0]+img_x.size[0], img_z.size[0]), \
            max(img_y.size[1]+img_z.size[1], img_x.size[1]+img_z.size[1])))

        img.paste(img_y, (0,0))
        img.paste(img_x, (img_y.size[0],0))
        img.paste(img_z, (0,max(img_y.size[1], img_x.size[1])))
        #img.show(); sys.exit(0)

        return img.convert("RGB")

    def colorSets(self, point_sets = [], volume_sets = []):
        """
        Adds colors to point_sets and volume_sets

        Input:
            point_sets = [points, ...]
            volume_sets = [mask, ...]

        Output:
            point_sets = [[points, colour=(255,0,0,100), outline=(0,0,0,255), size=5],...]
            volume_sets = [[mask, colour=(255,0,0,100)],...]
        """
        idx = 0 # color index

        vs = []
        for mask in volume_sets:
            vs.append([mask, self.getRGBA(idx, a=self.default_volume_alpha)])
            idx += 1

        ps = []
        for points in point_sets:
            ps.append([points, self.getRGBA(idx, a=self.default_point_alpha), \
                self.default_point_border, self.default_point_size])
            idx += 1

        return ps, vs

    def drawImageAutocolor(self, data3d, voxelsize, point_sets = [], volume_sets = []):
        """
        point_sets = [points, ...]
        volume_sets = [mask, ...]
        Returns RGB Image object

        Automatically adds colors to point_sets and volume_sets

        Save with: img.save(os.path.join(outputdir, "%s.png" % name))
        Open with: img.show()
        """
        ps, vs = self.colorSets(point_sets, volume_sets)
        return self.drawImage(data3d, voxelsize, point_sets=ps, volume_sets=vs)
