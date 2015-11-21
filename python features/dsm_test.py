from PIL import Image
import numpy as np
import scipy as sp
import cv2
from osgeo import osr, gdal
import skimage.io
import scipy.ndimage
from skimage import io
from matplotlib import pyplot as plt

im = skimage.io.imread('0_1_0_dsm.tif', plugin='tifffile')
threshold = 5
idx = im[:,:] < threshold
im[idx] = 0
#im2 = im[1000:2000,2000:3000]
im2 = im[5500:7500, 1500:3500]
# test 1000:2000, 2000:3000
# train 5500:7500, 1500:3500
#im3 = im[992:2008,1992:3008]
im3 = im[5492:7508, 1492:3508]
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
        box = im3[i-N+8:i+2*N+8,j-N+8:j+2*N+8]
        im_std[i/N,j/N] = np.std(box)
        #print im_std[i,j]

# Threshold
im4 = cv2.convertScaleAbs(im2)
print im4.dtype
print np.amax(im4)
im_thres = cv2.adaptiveThreshold(im4,155,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,61,0)
im_thres = scipy.ndimage.zoom(im_thres, 0.25, order=0) #downsample image
print im_thres.shape
#Load saved feature

feature_array = np.load("graylevels/f35_g128_b4_gau4_train.npy")
# print feature_array.shape
array_length = 250000 #train
#array_length = 62500 #test
im_std_array = np.reshape(im_std,[1,array_length])
im_std_array /= np.amax(im_std_array)
im_thres_array = np.reshape(im_thres,[1,array_length])
im_thres_array /= np.amax(im_thres_array)
# print im_std_array
feature_array = np.append(feature_array,im_std_array,axis=0)
feature_array = np.append(feature_array,im_thres_array,axis=0)
print feature_array.shape
#
np.save("f37_g128_b4_gau4_train.npy", feature_array)

#Check is reshaped image is correct
# ret_image = []
# ret_array = feature_array[-1,:]
# im_ret = np.reshape(ret_array,[250,250])
#print im_corr.shape

plt.figure()
io.imshow(im_std)
plt.figure()
io.imshow(im_thres)
plt.figure()
io.imshow(im2)
io.show()
