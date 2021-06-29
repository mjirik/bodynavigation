import h5py
import numpy as np
from loguru import logger
import random
import matplotlib.pyplot as plt
import lines
import CT_regression_tools

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import load_model

# from tensorflow.keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

# tf.debugging.set_log_device_placement(True)

size = 256

X_train = []
Y_train = []
validation = []
validation_y = []


with h5py.File(f"sagittal{size}.h5", "r") as h5f:
    for i in range(18):
        logger.info("Loading...")
        X_train.extend(np.asarray(h5f[f"scan_{i}"]))
        for j in range(len(np.asarray(h5f[f"scan_{i}"]))):
            Y_train.extend([np.asarray(h5f[f"label_{i}"])])
        logger.info(f"Scan {i+1} loaded for training")
    validation.extend(np.asarray(h5f[f"scan_{18}"]))
    for j in range(len(np.asarray(h5f[f"scan_{18}"]))):
        validation_y.extend([np.asarray(h5f[f"label_{18}"])])
    for i in range(20, 38):
        logger.info("Loading...")
        X_train.extend(np.asarray(h5f[f"scan_{i}"]))
        for j in range(len(np.asarray(h5f[f"scan_{i}"]))):
            Y_train.extend([np.asarray(h5f[f"label_{i}"])])
        logger.info(f"Scan {i+1} loaded for training")
    validation.extend(np.asarray(h5f[f"scan_{38}"]))
    for j in range(len(np.asarray(h5f[f"scan_{38}"]))):
        validation_y.extend([np.asarray(h5f[f"label_{38}"])])

X_train = np.asarray(X_train).reshape(np.asarray(X_train).shape[0], size, size, 1)
validation = np.asarray(validation).reshape(
    np.asarray(validation).shape[0], size, size, 1
)

# for i in range(5):
#     k = 150
#     k += 100
#     print(Y_train[k])
#     alpha = Y_train[k][0]
#     delta = Y_train[k][1]
#     line = lines.linesplit(alpha, delta, 256)
#     plt.imshow(X_train[k], cmap='gray')
#     plt.contour(line)
#     plt.show()


model = Sequential()

model.add(Convolution2D(32, 3, 3, activation="relu", input_shape=(size, size, 1)))
model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2))
model.summary()

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])
model.fit(
    X_train,
    np.asarray(Y_train),
    batch_size=32,
    epochs=50,
    validation_data=(validation, np.asarray(validation_y)),
    verbose=1,
)

model.save(f"simpleCNN{size}.h5")
