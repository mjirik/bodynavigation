import io3d
import sed3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io3d.datareaderqt
from loguru import logger
import os
import skimage.io
import skimage
import skimage.transform
import random
import h5py



def annotate(number_of_scans): #annotation starting from scan 1
    df = pd.DataFrame(columns = ['Ircad ID' , 'Mark 1 slice id', 'Mark 2 slice id' , 'Mark 3 slice id', 'Mark 4 slice id']) 

    for i in range(number_of_scans):
        dr = io3d.DataReader()
        # TODO nastavit cestu k datasetům
        # join_path přidá před zadanou cestu část vedoucí k uživatelskému datasetu.
        # To nám pomáhá spouštět kód na různých počítačích bez ohledu na to, kde jsou data stažená
        # Jen je potřeba editovat soubor v domovské složce uživatele ".io3d_cache.yaml"
        # na mém počítači je to tedy v C:\Users\Jirik\.io3d_cache.yaml
        # Do tohoto souboru napište, kde máte data stažená a mělo by to chodit. Lomítka používejte raději dopředná.
        pth = io3d.datasets.join_path('medical/orig/3Dircadb1.{}/'.format(i+1), get_root=True)
        datap = dr.Get3DData(pth + "PATIENT_DICOM/", dataplus_format=True)
        datap_labelled = dr.Get3DData(pth + 'MASKS_DICOM/liver', dataplus_format=True)

        ed = sed3.sed3(datap['data3d'], contour = datap_labelled['data3d'], windowW = 400, windowC = 40)
        ed.show()

        nz = np.nonzero(ed.seeds)
        ids = np.unique(nz[0])
        order = input("Did liver end before kidney started? (y/n)")
        if order == "y":
            df = df.append({'Ircad ID' : i+1,'Mark 1 slice id' : ids[0], 'Mark 2 slice id' : ids[1], 'Mark 3 slice id' : ids[2],'Mark 4 slice id' : ids[3]}, ignore_index = True)
        elif order == "n":
            df = df.append({'Ircad ID' : i+1,'Mark 1 slice id' : ids[0], 'Mark 2 slice id' : ids[2], 'Mark 3 slice id' : ids[1],'Mark 4 slice id' : ids[3]}, ignore_index = True)
        else:
            print("ERROR")
            break;

    #df.to_excel('tabulka.xlsx', sheet_name='List1', index = False)

def getsliceid(scannum, slicenum): # Ircad ID, index of wanted slice
    df = pd.read_excel('tabulka.xlsx', sheet_name='List1') # getting data from excel
    scan = df.iloc[scannum-1] # selecting the specific row from the table

    pth = io3d.datasets.join_path('medical/orig/3Dircadb1.{}/'.format(scannum), get_root=True)
    list = os.listdir(pth + '/PATIENT_DICOM/')
    total_slices = len(list)-1 # getting the total number of slices in this scan
    if slicenum > total_slices:
        raise ValueError('Slice ID is bigger than the number of slices in this scan.')
    
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
        corner2 = total_slices # getting corners and bases for labeling
    else:
        raise Error('Error getting slice label')

    #x = corner2 - corner1
    #y = 1 / x
    #z = base + y * (slicenum - corner1)
    return base + (1 / (corner2 - corner1)) * (slicenum - corner1)

def show(img):
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.show()

def loadscan(scannum):
    # TODO k číslu série se v jiné části kódu příčítá jednička, takže to pracuje s pacientem 2 a 3 zároveň
    #
    pth = io3d.datasets.join_path('medical/orig/3Dircadb1.{}/PATIENT_DICOM/'.format(scannum), get_root=True)
    datap = io3d.read(pth)
    data3d = datap['data3d']
    scan = []
    i = 1
    for slice in data3d:
        id = getsliceid(scannum, i)
        scan.append([slice, id])
        i += 1
    return scan # [slice, label] [slice, label] ...

def resize(img):
    img = skimage.transform.resize(np.asarray(img), [64,64], preserve_range = True)
    return img

def addnormal(img):
    scale = random.randint(50,150)
    img = img + np.random.normal(0, scale, [512, 512])
    return img

def rotate(img):
    angle = random.randint(1,4)
    direction = bool(random.getrandbits(1))
    if direction is False:
        angle *= -1
    #print(angle)
    img = skimage.transform.rotate(img, angle, cval = -1000, mode = 'constant', preserve_range = True)
    return img

def normalize(img):
    img = resize(img)
    img += 1000
    img = np.array(img)
    img = img * (1/2000)
    
    for row in img:
        for i in range(len(row)):
            if row[i] > 1:
                row[i] = 1
            elif row[i] < 0:
                row[i] = 0
    return img

def normalizescan(scan):
    normalized = np.empty((len(scan),), dtype=object)
    normalized[:] = [[] * len(normalized)]
    for i in range(len(scan)):
        normalized[i] = np.asarray([normalize(scan[i][0]), scan[i][1]])
    return normalized

def augmentscan(scan):
    augmented = []
    for slice in scan:
        augmented.append([slice[0], slice[1]]) # normal slice
        aug1 = addnormal(slice[0])
        augmented.append([aug1, slice[1]]) # filtered slice
        aug2 = rotate(slice[0])
        augmented.append([aug2, slice[1]]) # rotated slice
        aug3 = rotate(addnormal(slice[0]))
        augmented.append([aug3, slice[1]]) # rotated filtered slice
    return augmented

def save():
    # TODO co takhle počítat od jedné?
    # for i in range(1,21): # number of scans in the dataset
    for i in range(20): # number of scans in the dataset
        scan = loadscan(i+1)
        print("Scan {} loaded : {} slices".format(i+1,len(scan)))
    
        augmented = augmentscan(scan)
        print("Scan {} augmented: {} slices".format(i+1, len(augmented)))
    
        normalized = normalizescan(augmented)
        print("Scan {} normalized : {} slices".format(i+1,len(normalized)))
    
        labels = np.zeros(len(normalized))
        # slices = np.empty((len(normalized),), dtype=object)
        # slices[:] = [[] * len(slices)]
        sh = normalized[0][0].shape
        slices = np.empty([len(normalized), sh[0], sh[1]])

        for i in range(len(normalized)):
            labels[i] = normalized[i][1]
            slices[i] = np.asarray(normalized[i][0])
            slices[i, :,:] = np.asarray(normalized[i][0])

        with h5py.File('data.h5', 'w') as h5f:
            h5f.create_dataset('scan_{}'.format(i+1), data=slices)
            h5f.create_dataset('label_{}'.format(i+1), data=labels)
        print('saved')