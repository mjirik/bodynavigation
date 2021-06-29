import h5py
import numpy as np
from loguru import logger
from pathlib import Path

# import random
# import matplotlib.pyplot as plt
# import lines
# import CT_regression_tools
# import sed3

import tensorflow as tf

# import os
# from skimage.transform import resize
# from skimage.io import imsave
import numpy as np

# from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    concatenate,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History

# from skimage.exposure import rescale_intensity
# from skimage import io
# from data import load_train_data, load_test_data
# from sklearn.utils import class_weight


def get_unet(weights=None):
    if weights is None:
        weights = [0.05956, 3.11400]

    inputs = Input((256, 256, 1))
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    up6 = concatenate(
        [Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv5), conv4],
        axis=3,
    )
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate(
        [Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv6), conv3],
        axis=3,
    )
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate(
        [Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv7), conv2],
        axis=3,
    )
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv8), conv1],
        axis=3,
    )
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(1, (1, 1))(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

    return model


def train(
    imshape=256,
    # sdf_type='diaphragm_axial',
    sdf_type="coronal",
    # sdf_type='sagittal',
    # sdf_type='surface',
    skip_h5=False,
    batch_size=8,
    epochs=50,
    filename_prefix="",
    validation_ids=[19, 39],
    test_ids=[20, 40],
    n_data=40,
):
    """Train an U-net neural network with specified parameters.

    Args:
        imshape (int, optional): Shape of dataset's images. Defaults to 256.
        sdf_type (str, optional): Type of sdf method we train. Can be: sagittal, coronal, surface or diaphragm_axial.
        skip_h5 (bool, optional): Specify True if you want to skip saving the model. Defaults to False.
        batch_size (int, optional): Fitting batch size. Defaults to 16.
        epochs (int, optional): Number of epochs for fitting. Defaults to 50.
        filename_prefix (str, optional): Option to add a prefix to create new files when testing. Defaults to ''.
        validation_ids (list, optional): Validation data scans' indexes. Defaults to [19, 39].
        test_ids (list, optional): Test data scans' indexes that will be skipped when loading training data. Defaults to [20, 40].
        n_data (int, optional): Number of scans used. Defaults to 40. 40 is the maximum.

    Returns:
        model: keras h5 file that can be loaded by keras.models.load_model()
    """

    X_train = []
    Y_train = []
    validation = []
    validation_y = []

    pth = Path(__file__).parent

    # Data loading
    with h5py.File(pth / f"{filename_prefix}sdf_{sdf_type}{imshape}.h5", "r") as h5f:
        logger.debug(h5f.keys())
        for i in range(n_data):
            if i + 1 in validation_ids:
                validation.extend(np.asarray(h5f[f"scan_{i}"]))
                validation_y.extend(np.asarray(h5f[f"label_{i}"]))
                pass
            elif i + 1 in test_ids:
                pass
            else:
                logger.info("Loading...")
                X_train.extend(np.asarray(h5f[f"scan_{i}"]))
                Y_train.extend(np.asarray(h5f[f"label_{i}"]))
                logger.info(f"Scan {i+1} loaded for training")

    # sed3.show_slices(np.asarray(X_train[50:100]), np.asarray(Y_train[0:50]), slice_step=10, axis=0)
    # sed3.show_slices(np.asarray(X_train[100:150]), np.asarray(Y_train[0:50]), slice_step=10, axis=0)
    # sed3.show_slices(np.asarray(X_train[150:200]), np.asarray(Y_train[0:50]), slice_step=10, axis=0)
    # sed3.show_slices(np.asarray(X_train[200:250]), np.asarray(Y_train[0:50]), slice_step=10, axis=0)

    # plt.show()

    # Reshaping
    X_train = np.asarray(X_train).reshape(np.asarray(X_train).shape[0], 256, 256, 1)
    validation = np.asarray(validation).reshape(
        np.asarray(validation).shape[0], 256, 256, 1
    )

    # Reshaping label data
    Y_train = np.asarray(Y_train).reshape(np.asarray(Y_train).shape[0], 256, 256, 1)
    validation_y = np.asarray(validation_y).reshape(
        np.asarray(validation_y).shape[0], 256, 256, 1
    )

    logger.debug(f"train.shape = {Y_train.shape}")
    logger.debug(f"validation.shape = {validation.shape}")

    # Fitting
    model = get_unet()
    model.fit(
        X_train,
        np.asarray(Y_train),
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation, np.asarray(validation_y)),
        verbose=1,
    )

    # Saving the model to .h5 (optional)
    if not skip_h5:
        model.save(f"{filename_prefix}sdf_unet_{sdf_type}.h5")
        logger.info(f"Model saved as {filename_prefix}sdf_unet_{sdf_type}.h5")

    # return model


if __name__ == "__main__":
    # this will be skipped if file is imported but it will work if file is called from commandline
    train()
