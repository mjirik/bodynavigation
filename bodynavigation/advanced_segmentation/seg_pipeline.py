import numpy as np
from loguru import logger
import h5py
import skimage.io
import skimage
import skimage.transform
import CT_regression_tools
import seg
import lines

imshape = 256


c = 0

for i in range(40):
    if i <= 19:
        ss, data, voxelsize = seg.read_scan("3Dircadb1", i + 1)
    else:
        ss, data, voxelsize = seg.read_scan("sliver07", i - 19)

    X_train = [[] for j in range(len(data))]
    for j in range(len(data)):
        img = CT_regression_tools.resize(data[j], imshape)
        img = CT_regression_tools.normalize(img)
        X_train[j] = img

    sagittal = np.abs(ss.dist_sagittal()) > 0
    symmetry_point_mm = ss.symmetry_point_wvs * ss.voxelsize_mm[1:]
    alpha, delta = lines.normal_from_slopeintercept(ss.angle, symmetry_point_mm)

    alpha = 180 - alpha
    if alpha > 360:
        alpha -= 360
    if alpha < 0:
        alpha += 360
    Y_train = [alpha, round(float(delta), 4)]

    with h5py.File(f"sagittal{imshape}.h5", "a") as h5f:
        h5f.create_dataset("scan_{}".format(i), data=np.asarray(X_train))
        h5f.create_dataset("label_{}".format(i), data=Y_train)
    c += 1
    logger.info(f"Scan n.{c} saved.")
