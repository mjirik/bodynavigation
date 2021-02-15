import seg
import numpy as np
from loguru import logger
import h5py
import lines
import skimage.io
import skimage
import skimage.transform
import CT_regression_tools
import matplotlib.pyplot as plt
# from bodynavigation.organ_detection import OrganDetection
c=0
imshape = 256

for i in range(40):
    if i <= 19:
        ss, data, voxelsize = seg.read_scan("3Dircadb1", i+1)
    else:
        ss, data, voxelsize = seg.read_scan("sliver07", i-19)
    # logger.info("starting")
    X_train = [[] for j in range(len(data))]
    for j in range(len(data)):
        
            img = CT_regression_tools.resize(data[i], 256)
            img = CT_regression_tools.normalize(img)
            X_train[j] = img
    # logger.info("organdetection")
    # obj = OrganDetection(data, voxelsize)
    
    # logger.info("obj created")
    # Y_train = obj.getBody()
    Y_train = ss.dist_surface()
    Y_train = skimage.transform.resize(np.asarray(Y_train), [Y_train.shape[0], imshape, imshape], preserve_range = True)
    
    plt.imshow(X_train, cmap='gray')
    plt.contour(Y_train>0)
    plt.show()
    
    with h5py.File('sdf_diaphragm_axial256.h5', 'a') as h5f:
        h5f.create_dataset('scan_{}'.format(i), data=np.asarray(X_train))
        h5f.create_dataset('label_{}'.format(i), data=Y_train)
    c += 1
    logger.info(f'Scan n.{c} saved.')