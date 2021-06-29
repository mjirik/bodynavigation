import tensorflow as tf
import h5py
import numpy as np
from loguru import logger

# tf.keras.models.Model


size = 256


X_train = []
Y_train = []
validation = []
validation_y = []


with h5py.File(f"sagittal{size}s.h5", "r") as h5f:
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

model: tf.keras.models.Model = tf.keras.models.load_model(f"simpleCNNgpu{size}s.h5")
model.summary()


loss, acc = model.evaluate(validation, np.asarray(validation_y), verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
