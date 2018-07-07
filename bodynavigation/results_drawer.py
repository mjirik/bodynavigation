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

from .tools import firstNonzero, resize

class ResultsDrawer(object):
    """
    Draws composite image of segmentation results with views along each axis
    Usage:
        rd = ResultsDrawer()
        img = rd.drawImageAutocolor(data3d, voxelsize, volume_sets = masks)
        img.show()
    """

    COLORS = [
        (255,0,0), (255,106,0), (255,213,0), (191,255,0), (0,255,21), \
        (0,255,234), (0,170,255), (43,0,255), (255,0,255), (255,0,149)
        ]

    def __init__(self, data3d_forced_min = -1024, data3d_forced_max = 1024,\
            default_volume_alpha = 100, default_point_alpha = 255, \
            default_point_border = (0,0,0), default_point_size = 5, \
            mask_depth = False, mask_depth_sort = False, mask_depth_scale = 0.75 ):
        """
        mask_depth_sort - forced True if mask_depth=True
        """
        self.data3d_forced_min = data3d_forced_min
        self.data3d_forced_max = data3d_forced_max
        self.default_volume_alpha = default_volume_alpha
        self.default_point_alpha = default_point_alpha
        if default_point_border is None:
            self.default_point_border = None
        else:
            self.default_point_border = tuple(list(default_point_border)+[default_point_alpha,])
        self.default_point_size = default_point_size
        self.mask_depth = mask_depth
        self.mask_depth_scale = mask_depth_scale
        self.mask_depth_sort = True if mask_depth else mask_depth_sort

    def getRGBA(self, idx, a=None):
        """ Returns RGB/RGBA color with specified intex """
        while idx >= len(self.COLORS):
            idx = idx - len(self.COLORS)
        c = list(self.COLORS[idx])
        if a is not None: c.append(a)
        return tuple(c)

    def _drawPoints(self, img, points, colour=(255,0,0,255), outline=None, size=1):
        """
        Draws points into image (view)

        img - 2D array
        points - 2D coordinates
        """
        if len(points) == 0: return img
        if len(colour)==3:
            colour = (colour[0],colour[1],colour[2],self.default_point_alpha)
        if colour[3] == None:
            colour[3] = self.default_point_alpha

        img_d = Image.new('RGBA', img.size)
        draw = ImageDraw.Draw(img_d)

        # draw outline/border
        if outline is not None:
            bsize = size+2
            for p in points: # p = [x,y]
                xy = [p[0]-(bsize/2), p[1]-(bsize/2), p[0]+(bsize/2), p[1]+(bsize/2)]
                draw.rectangle(xy, fill=outline)

        # draw points
        for p in points: # p = [x,y]
            xy = [p[0]-(size/2), p[1]-(size/2), p[0]+(size/2), p[1]+(size/2)]
            draw.rectangle(xy, fill=colour)

        img = Image.composite(img_d, img, img_d)
        return img

    def _drawVolume(self, img, mask, colour=(255,0,0,150)):
        """
        Draws volume (mask) into image (view)

        img - 2D Image
        mask - 2D array; bool or int32 <0,1;256>
        """
        # get base colour
        if len(colour)==3:
            colour = (colour[0],colour[1],colour[2],self.default_volume_alpha)
        if colour[3] == None:
            colour[3] = self.default_volume_alpha

        mask = mask.astype(np.uint8) # fixes future numpy warning
        img_mask = np.zeros((mask.shape[0],mask.shape[1], 4), dtype=np.uint8)
        for d in range(np.max(1, mask.min()), mask.max()+1): # range(1,256+1)
            scale = 1.0 - (((d-1)/255.0) * self.mask_depth_scale )
            mask_d = (mask == d)
            img_mask[mask_d,0] = int(colour[0]*scale)
            img_mask[mask_d,1] = int(colour[1]*scale)
            img_mask[mask_d,2] = int(colour[2]*scale)

        img_mask[:,:,3] = (mask!=0).astype(np.uint8)*colour[3]
        img_mask = Image.fromarray(img_mask, 'RGBA')
        #img_mask.show()
        img = Image.composite(img_mask, img, img_mask)

        return img

    def _getView(self, data3d, voxelsize, axis=0, mask=False):
        """
        Returns raw view of data or mask along one axis

        data3d - only positive numbers or zeros
        axis - 0,1,2
        maks - set true if input is mask
        """
        if axis not in [0,1,2]: raise Exception("Wrong axis parameter: %s" % str(axis))

        if mask:
            # calculates depth <1;shape_max>
            view = firstNonzero(data3d, axis=axis, invalid_val=-1) + 1
            # scale depth to <1;256>; 0 is no mask
            view = np.ceil(view.astype(np.float)*(256.0/data3d.shape[axis])).astype(np.int32)
        else:
            # converts to <0;255>
            view = np.sum(data3d, axis=axis, dtype=np.int32).astype(np.float)
            view = (view*(255.0/view.max())).astype(np.int32)

        if axis == 0:
            new_shape = (int(data3d.shape[1] * voxelsize[1]), int(data3d.shape[2] * voxelsize[2]))
        elif axis == 1:
            new_shape = (int(data3d.shape[0] * voxelsize[0]), int(data3d.shape[1] * voxelsize[1]))
        else: # axis == 2
            new_shape = (int(data3d.shape[0] * voxelsize[0]), int(data3d.shape[2] * voxelsize[2]))

        order = 0 if mask else 1
        view = resize(view, new_shape, order=order, mode="reflect").astype(np.int32)

        return view

    def _getViewPoints(self, points, voxelsize, axis):
        """
        Converts raw 3D point coordinates to 2D view coordinates

        points - list of 3D coordinates
        voxelsize - (f,f,f)
        axis - 0,1,2
        """
        if axis not in [0,1,2]: raise Exception("Wrong axis parameter: %s" % str(axis))
        if len(points) == 0: return []

        points = [ list(np.asarray(p)*voxelsize) for p in points ]

        z, y, x = zip(*points)
        if axis == 0:
            points_2d = zip(x, y)
        elif axis == 1:
            points_2d = zip(x, z)
        elif axis == 2:
            points_2d = zip(y, z)

        return points_2d

    def _drawView(self, view, axis, point_sets = [], volume_sets = []):
        """
        Draws complete view along one axis

        view - numpy array <0;255>
        axis - 0,1,2
        point_sets - [[(points_z, colour=(255,0,0,100), outline=None, size=3),..],...]
        volume_sets - [[(mask_z, colour=(255,0,0,100)),...],...]

        Returns RGB Image object
        """
        if axis not in [0,1,2]: raise Exception("Wrong axis parameter: %s" % str(axis))

        img = Image.fromarray(view, 'I').convert("RGBA")
        for vset in volume_sets:
            mask, colour = tuple(vset[axis])
            img = self._drawVolume(img, mask, colour)
        for pset in point_sets:
            points, colour, outline, size = tuple(pset[axis])
            img = self._drawPoints(img, points, colour=colour, outline=outline, size=size)

        return img

    def _sortPixelDepthOfMasks(self, masks = []):
        """
        Deletes points in mask views that should be hidden behind points of another mask

        masks - [mask_z1,mask_z2,...]
        """
        if len(masks) == 0: return []

        # everything into one numpy array
        maska_np_shape = (len(masks), masks[0].shape[0], masks[0].shape[1])
        masks_np = np.zeros(maska_np_shape, dtype=masks[0].dtype)
        for i in range(len(masks)):
            masks_np[i,:,:] = masks[i]

        # delete all points that are not min along first axis
        fill_value = 10000
        masks_np[masks_np == 0] = fill_value
        for (x,y), min_idx in np.ndenumerate(np.argmin(masks_np, axis=0)):
            masks_np[:min_idx,x,y] = 0
            masks_np[(min_idx+1):,x,y] = 0
        masks_np[masks_np == fill_value] = 0

        # separate masks back into list
        masks = []
        for i in range(masks_np.shape[0]):
            masks.append(masks_np[i,:,:])

        return masks

    def drawImage(self, data3d, voxelsize, point_sets = [], volume_sets = []):
        """
        point_sets = [[points, colour=(255,0,0,100), outline=None, size=3],...]
        volume_sets = [[mask, colour=(255,0,0,100)],...]
        Returns RGB Image object

        Save with: img.save(os.path.join(outputdir, "%s.png" % name))
        Open with: img.show()
        """

        # prepare for brightness normalization
        data3d[ data3d < self.data3d_forced_min ] = self.data3d_forced_min
        data3d[ data3d > self.data3d_forced_max ] = self.data3d_forced_max
        data3d = data3d + abs(np.min(data3d))

        # axis views
        view_z = self._getView(data3d, voxelsize, axis=0)
        view_y = self._getView(data3d, voxelsize, axis=1)
        view_x = self._getView(data3d, voxelsize, axis=2)

        tmp = []
        for vset in volume_sets:
            mask, colour = tuple(vset)
            mask_z = self._getView(mask, voxelsize, axis=0, mask=True)
            mask_y = self._getView(mask, voxelsize, axis=1, mask=True)
            mask_x = self._getView(mask, voxelsize, axis=2, mask=True)
            if not self.mask_depth_sort:
                mask_z = mask_z != 0; mask_y = mask_y != 0; mask_x = mask_x != 0;
            tmp.append([ (mask_z, colour), (mask_y, colour), (mask_x, colour) ])
        volume_sets = tmp

        tmp = []
        for pset in point_sets:
            points, colour, outline, size = tuple(pset)
            points_z = self._getViewPoints(points, voxelsize, axis=0)
            points_y = self._getViewPoints(points, voxelsize, axis=1)
            points_x = self._getViewPoints(points, voxelsize, axis=2)
            tmp.append([
                (points_z, colour, outline, size),
                (points_y, colour, outline, size),
                (points_x, colour, outline, size)
                ])
        point_sets = tmp

        # sort depth of pixels in masks
        if self.mask_depth_sort:
            masks_z = []; masks_y = []; masks_x = [];
            for vset in volume_sets:
                masks_z.append(vset[0][0])
                masks_y.append(vset[1][0])
                masks_x.append(vset[2][0])

            masks_z = self._sortPixelDepthOfMasks(masks_z)
            masks_y = self._sortPixelDepthOfMasks(masks_y)
            masks_x = self._sortPixelDepthOfMasks(masks_x)

            tmp = []
            for i, vset in enumerate(volume_sets):
                tmp.append([
                    tuple([masks_z[i],]+list(vset[0][1:])),
                    tuple([masks_y[i],]+list(vset[1][1:])),
                    tuple([masks_x[i],]+list(vset[2][1:])),
                    ])
            volume_sets = tmp

        # draw views
        img_z = self._drawView(view_z, axis=0, point_sets=point_sets, volume_sets=volume_sets)
        img_y = self._drawView(view_y, axis=1, point_sets=point_sets, volume_sets=volume_sets)
        img_x = self._drawView(view_x, axis=2, point_sets=point_sets, volume_sets=volume_sets)

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
