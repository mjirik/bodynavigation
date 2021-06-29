import numpy as np
import io3d
import sed3
import io3d.datareaderqt
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import os
import skimage.io
import skimage
import skimage.transform
import random
import h5py
import tensorflow as tf
from tensorflow import keras
import SimpleITK as sitk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import load_model

# from tensorflow.keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from pathlib import Path


def annotate(number_of_scans):  # annotation starting from scan 1
    """
    Annotate the data form 3DIrcad1 dataset, starting from scan 1, up to scan passed in param.
    
    Save the labels to an excel file 'tabulka.xlsx'.
    """
    df = pd.DataFrame(
        columns=[
            "Ircad ID",
            "Mark 1 slice id",
            "Mark 2 slice id",
            "Mark 3 slice id",
            "Mark 4 slice id",
        ]
    )

    for i in range(number_of_scans):
        dr = io3d.DataReader()
        pth = io3d.datasets.join_path(
            "medical/orig/3Dircadb1.{}/".format(i + 1), get_root=True
        )
        datap = dr.Get3DData(pth + "PATIENT_DICOM/", dataplus_format=True)
        datap_labelled = dr.Get3DData(pth + "MASKS_DICOM/liver", dataplus_format=True)

        ed = sed3.sed3(
            datap["data3d"], contour=datap_labelled["data3d"], windowW=400, windowC=40
        )
        ed.show()

        nz = np.nonzero(ed.seeds)
        ids = np.unique(nz[0])
        order = input("Did liver end before kidney started? (y/n)")
        if order == "y":
            df = df.append(
                {
                    "Ircad ID": i + 1,
                    "Mark 1 slice id": ids[0],
                    "Mark 2 slice id": ids[1],
                    "Mark 3 slice id": ids[2],
                    "Mark 4 slice id": ids[3],
                },
                ignore_index=True,
            )
        elif order == "n":
            df = df.append(
                {
                    "Ircad ID": i + 1,
                    "Mark 1 slice id": ids[0],
                    "Mark 2 slice id": ids[2],
                    "Mark 3 slice id": ids[1],
                    "Mark 4 slice id": ids[3],
                },
                ignore_index=True,
            )
        else:
            print("ERROR")
            break
    df.to_excel("tabulka.xlsx", sheet_name="List1", index=False)


def annotate2(number_of_scans):  # annotation starting from scan 1
    """
    Annotate the data form our medical dataset, starting from scan 1, up to scan passed in param.
    
    Save the labels to an excel file 'tabulka2.xlsx'.
    """
    df = pd.DataFrame(
        columns=[
            "ID",
            "Mark 1 slice id",
            "Mark 2 slice id",
            "Mark 3 slice id",
            "Mark 4 slice id",
        ]
    )

    for i in range(number_of_scans):
        print(f"Showing scan n. {str(i+1).zfill(2)}")
        dr = io3d.DataReader()

        datap = dr.Get3DData(
            f"data/medical/orig/sliver07/training/liver-orig0{str(i+1).zfill(2)}.mhd",
            dataplus_format=True,
        )

        ed = sed3.sed3(datap["data3d"][::-1, :, :], windowW=400, windowC=40)
        ed.show()

        nz = np.nonzero(ed.seeds)

        ids = np.unique(nz[0])
        df = df.append(
            {
                "ID": i + 1,
                "Mark 1 slice id": ids[0],
                "Mark 2 slice id": ids[1],
                "Mark 3 slice id": ids[2],
                "Mark 4 slice id": ids[3],
            },
            ignore_index=True,
        )
    df.to_excel("tabulka2.xlsx", sheet_name="List1", index=False)


def getsliceid(scannum, slicenum):
    """ 
    Get the label of a scan slice from xls table (from Ircad dataset).
    
    Parameters

    scannum - Ircad ID
    slicenum - index of wanted slice

    ----
    returns the label as a float value
    """
    df = pd.read_excel("tabulka.xlsx", sheet_name="List1")  # getting data from excel
    scan = df.iloc[scannum - 1]  # selecting the specific row from the table

    pth = io3d.datasets.join_path(
        "medical/orig/3Dircadb1.{}/".format(scannum), get_root=True
    )
    list = os.listdir(pth + "/PATIENT_DICOM/")
    total_slices = len(list) - 1  # getting the total number of slices in this scan
    if slicenum > total_slices:
        raise ValueError("Slice ID is bigger than the number of slices in this scan.")

    if slicenum == scan[1]:
        return 1
    elif slicenum == scan[2]:
        return 2
    elif slicenum == scan[3]:
        return 3
    elif slicenum == scan[4]:
        return 4
    elif slicenum == 1:
        return 0
    elif slicenum == total_slices:
        return 5
    elif slicenum < scan[1]:
        base = 0
        corner1 = 1
        corner2 = scan[1]
    elif slicenum > scan[1] and slicenum < scan[2]:
        base = 1
        corner1 = scan[1]
        corner2 = scan[2]
    elif slicenum > scan[2] and slicenum < scan[3]:
        base = 2
        corner1 = scan[2]
        corner2 = scan[3]
    elif slicenum > scan[3] and slicenum < scan[4]:
        base = 3
        corner1 = scan[3]
        corner2 = scan[4]
    elif slicenum > scan[4]:
        base = 4
        corner1 = scan[4]
        corner2 = total_slices  # getting corners and bases for labeling
    else:
        raise Exception("Error getting slice label")
    return base + (1 / (corner2 - corner1)) * (slicenum - corner1)


def getsliceid2(scannum, slicenum):
    """ 
    Get the label of a scan slice from xls table (from second dataset).
    
    Parameters

    scannum - Ircad ID
    slicenum - index of wanted slice

    ----
    returns the label as a float value
    """
    df = pd.read_excel("tabulka2.xlsx", sheet_name="List1")  # getting data from excel
    scan = df.iloc[scannum - 1]  # selecting the specific row from the table

    with open(
        f"data/medical/orig/sliver07/training/liver-orig0{str(scannum).zfill(2)}.mhd",
        "r",
    ) as f:
        lines = f.read().splitlines()
        shape = lines[5].split(" ")[
            -1
        ]  # getting the total number of slices in this scan from .mhd file
    total_slices = int(shape)

    if slicenum > total_slices:
        raise ValueError("Slice ID is bigger than the number of slices in this scan.")

    if slicenum == scan[1]:
        return 1
    elif slicenum == scan[2]:
        return 2
    elif slicenum == scan[3]:
        return 3
    elif slicenum == scan[4]:
        return 4
    elif slicenum == 1:
        return 0
    elif slicenum == total_slices:
        return 5
    elif slicenum < scan[1]:
        base = 0
        corner1 = 1
        corner2 = scan[1]
    elif slicenum > scan[1] and slicenum < scan[2]:
        base = 1
        corner1 = scan[1]
        corner2 = scan[2]
    elif slicenum > scan[2] and slicenum < scan[3]:
        base = 2
        corner1 = scan[2]
        corner2 = scan[3]
    elif slicenum > scan[3] and slicenum < scan[4]:
        base = 3
        corner1 = scan[3]
        corner2 = scan[4]
    elif slicenum > scan[4]:
        base = 4
        corner1 = scan[4]
        corner2 = total_slices  # getting corners and bases for labeling
    else:
        raise Exception("Error getting slice label")
    return base + (1 / (corner2 - corner1)) * (slicenum - corner1)


def show(img):
    """
    Visualize a single slice passed as an x*y array.

    ----
    Returns void
    """
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.show()


def loadscan(scannum):
    """
    Load a scan along with its labels (from Ircad dataset).
    
    Parameters
    scannum - id of the loaded scan

    ----
    returns [[slice, label], [slice, label], ...]
    """
    pth = io3d.datasets.join_path(
        "medical/orig/3Dircadb1.{}/PATIENT_DICOM/".format(scannum), get_root=True
    )
    datap = io3d.read(pth)
    data3d = datap["data3d"]
    scan = []
    i = 1
    for slice in data3d:
        id = getsliceid(scannum, i)
        scan.append([slice, id])
        i += 1
    return scan


def loadscan2(scannum):
    """
    Load a scan along with its labels (from second dataset).
    
    Parameters
    scannum - id of the loaded scan

    ----
    returns [[slice, label], [slice, label], ...]
    """
    datap = io3d.read(
        f"data/medical/orig/sliver07/training/liver-orig0{str(scannum).zfill(2)}.mhd"
    )
    data3d = datap["data3d"]
    scan = []
    i = 1
    for slice in data3d:
        id = getsliceid2(scannum, i)
        scan.append([slice, id])
        i += 1
    return scan


def resize(img, shape):
    """
    Resize an image to a square shape.
    """
    img = skimage.transform.resize(
        np.asarray(img), [shape, shape], preserve_range=True, mode="reflect"
    )
    return img


def addnormal(img):
    """
    Add random normal filter to the image.
    """
    scale = random.randint(50, 150)
    img = img + np.random.normal(0, scale, [512, 512])
    return img


def rotate(img):
    """
    Rotate the image from -4 to +4 degrees at random.
    """
    angle = random.randint(1, 4)
    direction = bool(random.getrandbits(1))
    if direction is False:
        angle *= -1
    # print(angle)
    img = skimage.transform.rotate(
        img, angle, cval=-1000, mode="constant", preserve_range=True
    )
    return img


def normalize(img):
    """
    Normalize a scan slice.

    ----
    add 1000, divide by 2000, normalize to values between 0 and 1, cutting all excesses
    """
    img += 1000
    img = np.array(img)
    img = img * (1 / 2000)

    for row in img:
        for i in range(len(row)):
            if row[i] > 1:
                row[i] = 1
            elif row[i] < 0:
                row[i] = 0
    return img


def normalizescan(scan):
    """
    Use the fcn.normalize function on all slices of a scan.
    Returns the whole scan back.
    """
    normalized = np.empty((len(scan),), dtype=object)
    normalized[:] = [[] * len(normalized)]
    for i in range(len(scan)):
        normalized[i] = np.asarray([normalize(scan[i][0]), scan[i][1]])
    return normalized


def augmentscan(scan):
    """
    Augment the scan dataset, adding 3 augmented slices for each original slice:

    1 rotated
    1 filtered
    1 rotated and filtered
    """
    augmented = []
    for slice in scan:
        augmented.append([slice[0], slice[1]])  # normal slice
        aug1 = addnormal(slice[0])
        augmented.append([aug1, slice[1]])  # filtered slice
        aug2 = rotate(slice[0])
        augmented.append([aug2, slice[1]])  # rotated slice
        aug3 = rotate(addnormal(slice[0]))
        augmented.append([aug3, slice[1]])  # rotated filtered slice
    return augmented


def save():
    """
    Completely prepare the data for usage in keras models.
    
    Including:
    1) Extraction from original directories
    2) Pairing slices with their labels
    3) Augmenting all scans, quadrupling the total number of slices.
    4) Normalizing all data
    5) Saving all scans to a single .h5 file with a separate datasets for images and their labels
    """
    for i in range(1, 21):  # number of scans in the dataset
        scan = loadscan(i)
        logger.info(f"Scan {i} loaded : {len(scan)} slices")

        augmented = augmentscan(scan)
        logger.info(f"Scan {i} augmented: {len(augmented)} slices")

        normalized = normalizescan(augmented)
        logger.info(f"Scan {i} normalized : {len(normalized)} slices")

        labels = np.zeros(len(normalized))
        sh = normalized[0][0].shape
        slices = np.empty([len(normalized), sh[0], sh[1]])

        for j in range(len(normalized)):
            labels[j] = normalized[j][1]
            slices[j] = np.asarray(normalized[j][0])
            slices[j, :, :] = np.asarray(normalized[j][0])

        with h5py.File("data.h5", "a") as h5f:
            h5f.create_dataset("scan_{}".format(i), data=slices)
            h5f.create_dataset("label_{}".format(i), data=labels)
        logger.info("Scan saved")


def save2():
    """
    Completely prepare the data for usage in keras models.
    
    Including:
    1) Extraction from original directory.
    2) Pairing slices with their labels.
    3) Augmenting all scans, quadrupling the total number of slices.
    4) Normalizing all data
    5) Saving all scans to a single .h5 file with a separate datasets for images and their labels
    """
    for i in range(1, 21):  # number of scans in the dataset
        scan = loadscan2(i)
        logger.info(f"Scan {i} loaded : {len(scan)} slices")

        augmented = augmentscan(scan)
        logger.info(f"Scan {i} augmented: {len(augmented)} slices")

        normalized = normalizescan(augmented)
        logger.info(f"Scan {i} normalized : {len(normalized)} slices")

        labels = np.zeros(len(normalized))
        sh = normalized[0][0].shape
        slices = np.empty([len(normalized), sh[0], sh[1]])

        for j in range(len(normalized)):
            labels[j] = normalized[j][1]
            slices[j] = np.asarray(normalized[j][0])
            slices[j, :, :] = np.asarray(normalized[j][0])

        with h5py.File("data2.h5", "a") as h5f:
            h5f.create_dataset("scan_{}".format(i), data=slices)
            h5f.create_dataset("label_{}".format(i), data=labels)
        logger.info("Scan saved")


def loadfromh5(dataset, first, last):
    """
    Load a selected set scans from a .h5 file to the workspace.

    Parameters
    dataset = 1 = Ircad, 2 = medical
    first = first scan you want to load
    last = last scan you want to load

    ----
    returns X_train and Y_train lists for keras
    """
    X_train = []
    Y_train = []

    pth = Path(__file__).parent

    with h5py.File(pth / f"data{dataset}.h5", "r") as h5f:
        for i in range(first, last + 1):
            logger.info("Loading...")
            X_train.extend(np.asarray(h5f["scan_{}".format(i)]))
            Y_train.extend(np.asarray(h5f["label_{}".format(i)]))
            logger.info("Scan {} loaded".format(i))
    return X_train, Y_train


def eval(model, X_test, Y_test):
    """
    Evaluate the prediction results of the model.

    Parameters
    model - pretrained keras model
    X_test, Y_test - data for testing the model
    """
    predictions = model.predict(X_test, batch_size=500)
    dif = []
    for i in range(len(predictions)):
        logger.info(f"Prediction {i}:{predictions[i]}")
        logger.info(f"Truth: {Y_test[i]}")
        dif.append(abs(predictions[i] - Y_test[i]))
        logger.info(f"Error: {dif[i]}")
        # if dif[i] > 1:
        # show(X_test[i].squeeze()) #show any slices, where prediction error reached 1
    logger.info(f"Average error = {sum(dif)/len(dif)}")
    plt.plot(predictions, label="Predictions")
    plt.plot(Y_test, label="Truth")
    plt.plot(dif, label="error")
    plt.legend()
    plt.show()


def modelcreation1(dataset, fromscan, toscan):
    """
    Creates a convolutional keras neural network, training it with data from ct scans dataset.

    Parameters
    dataset - 1 = Ircad, 2 = medical
    fromscan - the first scan used to train the model
    toscan - last scan used to train the model

    ----
    Returns the model
    """

    X_train, Y_train = loadfromh5(dataset, fromscan, toscan)
    X_train = np.asarray(X_train).reshape(np.asarray(X_train).shape[0], 64, 64, 1)

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation="relu", input_shape=(64, 64, 1)))
    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.summary()

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
    return model


def modelcreation2():
    """
    Creates a convolutional keras neural network, training it with data from ct scans from both datasets.

    ----
    Returns the model
    """

    X_train, Y_train = loadfromh5(1, 2, 19)
    X_train1, Y_train1 = loadfromh5(2, 2, 19)
    # print(f'{len(X_train)} + {len(X_train1)} = {len(X_train)+len(X_train1)}')
    # print(f'{len(Y_train)} + {len(Y_train1)} = {len(Y_train)+len(Y_train1)}')

    X_train.extend(X_train1)
    Y_train.extend(Y_train1)
    # print(f'{len(X_train)}')
    # print(f'{len(Y_train)}')

    X_train = np.asarray(X_train).reshape(np.asarray(X_train).shape[0], 64, 64, 1)

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation="relu", input_shape=(64, 64, 1)))
    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.summary()

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
    return model


def VGG_16(weights_path=None):
    """
    Creates a convolutional keras neural network, training it with data from ct scans from both datasets.
    Using the VGG-16 architecture.

    ----
    Returns the model
    """

    X_train, Y_train = loadfromh5(1, 2, 19)
    X_train1, Y_train1 = loadfromh5(2, 2, 19)

    X_train.extend(X_train1)
    Y_train.extend(Y_train1)

    X_train = np.asarray(X_train).reshape(np.asarray(X_train).shape[0], 64, 64, 1)
    # X_train = np.transpose(X_train, (0,3,1,2))

    print(X_train.shape)

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(64, 64, 1)))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["mse"])
    K.set_value(model.optimizer.learning_rate, 0.001)
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

    if weights_path:
        model.load_weights(weights_path)

    return model
