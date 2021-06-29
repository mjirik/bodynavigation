import h5py
import numpy as np
from loguru import logger
import random
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import load_model
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K

model = load_model("simpleCNNgpu256s.h5")
testscan = 19  # number of scan to be predicted


pred = []
truth = []

for i in range(40):
    X_test = []
    Y_test = []
    with h5py.File("sagittal256s.h5", "r") as h5f:
        logger.info("Loading...")
        X_test.extend(np.asarray(h5f[f"scan_{i}"]))
        for j in range(len(np.asarray(h5f[f"scan_{i}"]))):
            Y_test.extend([np.asarray(h5f[f"label_{i}"])])
        logger.info(f"Scan {i+1} loaded")

    X_test = np.asarray(X_test).reshape(np.asarray(X_test).shape[0], 256, 256, 1)

    predictions = model.predict(X_test)
    logger.info(f"Prediction {i}:{predictions[0]}")
    logger.info(f"Truth: {Y_test[0]}")
    pred.append(predictions[0])
    truth.append(Y_test[0])


plt.plot(pred, "o", label="Predictions")
plt.plot(truth, "o", label="Truth")
plt.legend()
plt.show()
