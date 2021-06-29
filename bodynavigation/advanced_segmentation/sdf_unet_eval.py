import h5py
import numpy as np
from loguru import logger
import sed3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


def test(
    imshape=256,
    # sdf_type='diaphragm_axial',
    # sdf_type='coronal',
    sdf_type="sagittal",
    # sdf_type='surface',
    filename_prefix="",
    test_ids=[20, 40],
):
    """Evaluates a U-net NN with test data loaded from a .h5 file.

    Args:
        imshape (int, optional): Shape of dataset's images. Defaults to 256.
        sdf_type (str, optional): Type of sdf method we train. Can be: sagittal, coronal, surface or diaphragm_axial.
        filename_prefix (str, optional): Option to add a prefix to create new files when testing. Defaults to ''.
        test_ids (list, optional): Test data scans' indexes in the dataset. Defaults to [20, 40].
    """

    model = load_model(f"{filename_prefix}sdf_unet_{sdf_type}.h5")

    for i in range(len(test_ids)):
        X_test = []
        Y_test = []

        # Loading test data
        with h5py.File(f"{filename_prefix}sdf_{sdf_type}{imshape}.h5", "r") as h5f:
            logger.info("Loading...")
            X_test.extend(np.asarray(h5f[f"scan_{i}"]))
            Y_test.extend(np.asarray(h5f[f"label_{i}"]))
            logger.info(f"Scan {test_ids[i]} loaded")

        # Reshaping
        X_test = np.asarray(X_test).reshape(np.asarray(X_test).shape[0], 256, 256, 1)
        Y_test = np.asarray(Y_test).reshape(np.asarray(Y_test).shape[0], 256, 256, 1)

        # Predictions
        predictions = model.predict(X_test)

        # Visualization
        X_test = np.asarray(X_test).reshape(np.asarray(X_test).shape[0], 256, 256)
        predictions = np.asarray(predictions).reshape(
            np.asarray(predictions).shape[0], 256, 256
        )
        Y_test = np.asarray(Y_test).reshape(np.asarray(Y_test).shape[0], 256, 256)

        sed3.show_slices(
            np.asarray(X_test[0:-1]), np.asarray(Y_test[0:-1]), slice_step=10, axis=0
        )
        sed3.show_slices(
            np.asarray(X_test[0:-1]),
            np.asarray(predictions[0:-1]),
            slice_step=10,
            axis=0,
        )


if __name__ == "__main__":
    # this will be skipped if file is imported but it will work if file is called from commandline
    test()
