from PIL import Image
import numpy as np
import scipy as sp
import cv2
from osgeo import osr, gdal
import skimage.io
from skimage import io
from matplotlib import pyplot as plt

im = skimage.io.imread('0_1_0_dsm.tif', plugin='tifffile')
threshold = 5
idx = im[:,:] < threshold
im[idx] = 0
im2 = im[1000:3000,1000:3000]
im3 = im[992:3008,992:3008]
print im2.shape[0]
#np.set_printoptions(threshold = np.nan)
#im2 = np.floor(im2)
#print np.unique(im2)
#print im
N = 4 #blockssize
im_std = np.zeros((im2.shape[0]/N,im2.shape[1]/N))
print im_std.shape

for i in range(0,im2.shape[0],N):
    print i
    for j in range(0,im2.shape[1],N):
        #print j'
        box = im3[i-N:i+2*N,j-N:j+2*N]
        im_std[i/N,j/N] = np.std(box)
        #print im_std[i,j]

#Load saved feature
feature_array = np.load("code/python features/graylevels/f16_g128_b4_train.npy")
print feature_array.shape
im_std_array = np.reshape(im_std,[1,250000])
print im_std_array.shape
feature_array = np.append(feature_array,im_std_array,axis=0)
print feature_array.shape

np.save("f16_g128_b4_train.npy", feature_array)
#Check is reshaped image is correct
# ret_image = []
# ret_array = feature_array[16,:]
# im_ret = np.reshape(ret_array,[500,500])
#print im_corr.shape

plt.figure()
io.imshow(im_std)
plt.figure()
io.imshow(im2)
io.show()
