from PIL import Image
import numpy as np
import scipy as sp
import cv2
from osgeo import osr, gdal
import skimage.io
import scipy.ndimage
from skimage import io
from matplotlib import pyplot as plt

'''Generates adaptive threshold feature. '''
'''Called from features.py with divide='none'. If used to generate height feature'''
'''seperatly, call with divide='one' or divide='multiple'.'''

def heightFeatures(image, block_size, divide):
	im = skimage.io.imread(image, plugin='tifffile')

	threshold = 0 #Remove negative values (generally -1024)
	idx = im[:,:] < threshold
	im[idx] = 0
	kernel_size = 101 #for threshold, trial and error to get good results
	im_xsize = im.shape[0]
	im_ysize = im.shape[1]

	if (divide == 'multiple'): #used to generate multiple cut out images
		# Variables
		cutout_xmin2 = 5500
		cutout_ymin2 = 1500
		cutout_xsize = 2000
		cutout_ysize = 2000
		x_range = int(np.floor(im_xsize/cutout_xsize)*cutout_xsize)
		y_range = int(np.floor(im_ysize/cutout_ysize)*cutout_ysize)

		counter = 1
		for j in range(0,y_range, int(cutout_ysize)):
			cutout_xmin = j
			for k in range(0,x_range, int(cutout_xsize)):
				cutout_ymin = k
				c = str(counter)
				cutout = im[k:k+cutout_xsize,j:j+cutout_ysize]
				cutout = cv2.convertScaleAbs(cutout)
				# Calc adaptive threshold
				cutout_thres = cv2.adaptiveThreshold(cutout,np.amax(cutout),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,kernel_size,0)
				cutout_thres = scipy.ndimage.zoom(cutout_thres, 1/float(N), order=0) #downsample image
				# Reshape the array
				array_length = cutout_thres.size
				cutout_thres_array = np.reshape(cutout_thres,[1,array_length])
				cutout_thres_array /= np.amax(cutout_thres_array)
				np.save(str(output_filename + str(counter) + '.npy'), cutout_thres_array)
				#cv2.imwrite(str(output_filename + str(counter) + '.png'), cutout_thres)
				counter = counter +1
				print counter
	else: #used to generate one image
		if(divide=='one'): #if a cutout is to be made
			im = im[cutout_xmin2:(cutout_xmin2+cutout_xsize), cutout_ymin2:(cutout_ymin2+cutout_ysize)]
		# Threshold
		im = cv2.convertScaleAbs(im)
		im_thres = cv2.adaptiveThreshold(im,np.amax(im),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,kernel_size,0)
		im_thres = scipy.ndimage.zoom(im_thres, 1/float(block_size), order=0) #downsample image
		np.set_printoptions(threshold = np.nan)
		array_length = im_thres.size
		im_thres_array = np.reshape(im_thres,[1,array_length])
		im_thres_array /= np.amax(im_thres_array)
		return im_thres_array
		#np.save(str(output_filename + '.npy'), im_thres_array)
		#plt.figure()
		#io.imshow(im_thres)
		#io.show()

#N = 4 #blockssize #for std calc, same as other features
#image = '0_1_0_dsm.tif' #input dsm image, must be tif
#output_name = 'output_dsm' #output name, npy format

# use 'none', 'one' or 'multple' as last argument
#heightFeatures(image, output_name, 'none')
