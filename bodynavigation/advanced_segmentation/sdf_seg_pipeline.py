from bodynavigation.advanced_segmentation import seg
import numpy as np
from loguru import logger
import h5py
import sed3
from bodynavigation.advanced_segmentation import lines
import skimage.io
import skimage
import skimage.transform
from bodynavigation.advanced_segmentation import CT_regression_tools
import matplotlib.pyplot as plt

# from bodynavigation.organ_detection import OrganDetection


def prepare_data(
    imshape=256,
    sdf_type="diaphragm_axial",
    # sdf_type='coronal',
    # sdf_type='sagittal',
    # sdf_type='surface',
    skip_h5=False,
    n_data=40,
    filename_prefix="",
):
    """

    :param imshape:
    :param sdf_type:
    :param skip_h5:
    :param n_data:
    :param filename_prefix: used to prevent rewriting the files during testing
    :return:
    """

    c = 0
    for i in range(n_data):
        if i <= 19:
            ss, data, voxelsize = seg.read_scan("3Dircadb1", i + 1)
        else:
            ss, data, voxelsize = seg.read_scan("sliver07", i - 19)

        X_train = np.empty(
            [len(data), imshape, imshape], dtype=np.float
        )  # more efficient

        # for j in range(n_data):
        for j in range(data.shape[0]):

            img = CT_regression_tools.resize(data[j], imshape)
            img = CT_regression_tools.normalize(img)
            X_train[j] = img

        Y_train = eval(f"ss.dist_to_{sdf_type}()")
        Y_train = skimage.transform.resize(
            np.asarray(Y_train),
            [Y_train.shape[0], imshape, imshape],
            preserve_range=True,
        )

        # sed3.show_slices(np.asarray(X_train[0:50]), np.asarray(Y_train[0:50]), slice_step=10, axis=2)
        # plt.show()

        if not skip_h5:
            with h5py.File(f"{filename_prefix}sdf_{sdf_type}{imshape}.h5", "a") as h5f:
                logger.debug(f"X_train={X_train.dtype}")
                h5f.create_dataset(f"scan_{i}", data=X_train)
                h5f.create_dataset(f"label_{i}", data=Y_train)
            c += 1
            logger.info(f"Scan n.{c} saved. i={i}")


if __name__ == "__main__":
    # this will be skipped if file is imported but it will work if file is called from commandline
    prepare_data()
