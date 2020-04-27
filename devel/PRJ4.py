import io3d
import sed3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io3d.datareaderqt
from loguru import logger
import fcn
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential
import h5py

scan = fcn.loadscan(3)
fcn.normalize(scan[5][0])

fcn.save()

#model = Sequential()
 
#model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(64,64,1)))
#model.add(Convolution2D(32, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
 
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1))
#model.summary()

#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['error'])
#model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)