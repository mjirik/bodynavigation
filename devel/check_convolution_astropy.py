import astropy.convolution as ap
import numpy as np
from loguru import logger

kernel_size_mm = [150, 150]
working_vs = [1,1,1]

# kernel_size must be odd - lichý
kernel_size = np.asarray(kernel_size_mm) / working_vs[1:]
# print 'ks1 ', kernel_size
odd = kernel_size % 2
kernel_size = kernel_size + 1 - odd
# print 'ks2 ', kernel_size

# metoda 1
kernel = np.ones(np.round(kernel_size).astype(np.int))
kernel = kernel / (1.0 * np.prod(kernel_size))

a = np.array([0,0,0,0,1,1,1,1])
b = np.array([1,1, 1]) / 3



resb = ap.convolve(a, b)
resa = np.convolve(a, b)

logger.debug(resa)
logger.debug(resb)
assert np.array_equal(resa, resb)
