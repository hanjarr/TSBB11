import cv2
import numpy as np
import scipy
import skimage
import scipy.ndimage.filters as sf
from skimage.feature import greycomatrix, greycoprops
from PIL import Image
import utils
import heightFeatures
global levels, N, gaussNr

np.set_printoptions(threshold = np.nan)

def imStats(im_array,levels):
	array_size=np.shape(im_array)[2]

	measures = ['homogeneity', 'energy','dissimilarity','ASM']
	greyco_array=np.zeros((len(measures),array_size))

	for i in range(0,array_size):

		image_block=im_array[:,:,i]
		pixel_pairs = greycomatrix(image_block, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4 ], levels, normed = False)
		pixel_pairs = np.array(pixel_pairs,dtype=np.float)
		numberOPP = [np.sum(pixel_pairs[:,:,:,0]), np.sum(pixel_pairs[:,:,:,1]), np.sum(pixel_pairs[:,:,:,2]), np.sum(pixel_pairs[:,:,:,3])]

		weightingOPP = np.zeros((levels,levels,1 ,1))
		weightingOPP[:,:,:,0] = pixel_pairs[:,:,:,0]/numberOPP[0] + pixel_pairs[:,:,:,1]/numberOPP[1] + pixel_pairs[:,:,:,2]/numberOPP[2] + pixel_pairs[:,:,:,3]/numberOPP[3]

		feature_arrays = np.zeros((len(measures)))
		for index,measure in enumerate(measures):
			feature_arrays[index] = greycoprops(weightingOPP,measure)

		greyco_array[:,i]=feature_arrays
		
		print float(i)/float(array_size)

	for j in range(0,array_size):
		greyco_array[:,j]/=np.amax(greyco_array[:,j])
	
	return greyco_array


def gaussFilt(im,filt_const):

	im[:,:,0] = sf.gaussian_filter(im[:,:,0], filt_const)
	im[:,:,1] = sf.gaussian_filter(im[:,:,1], filt_const)
	im[:,:,2] = sf.gaussian_filter(im[:,:,2], filt_const)

	return im

def block_mean(im_array):

	array_size=np.shape(im_array)[2]
	mean_array=np.zeros((1,array_size))
	for i in range(0,array_size):
		mean_array[0,i]=im_array[:,:,i].mean()
	mean_array/=np.amax(mean_array[:])
	return mean_array

def featureExt(filename):

	num_features=0
	div=256/levels
	im_height="../images/divided images/vricon_dsm" + filename +".tif"

	# Import the images
	im_blue = cv2.imread("../images/divided images/vricon_ortho_blue" + filename +".png")
	im_red = cv2.imread("../images/divided images/vricon_ortho_red" + filename +".png")
	im_green = cv2.imread("../images/divided images/vricon_ortho_green" + filename +".png")
	im_nir = cv2.imread("../images/divided images/vricon_ortho_nir" + filename +".png")
	im_pan=cv2.imread("../images/divided images/vricon_ortho_pan" + filename +".png")

	num_blocks=np.shape(im_blue)[0]**2/(N**2)


	#mean and graylevels
	imageB = np.floor((im_blue[:,:,0]+im_blue[:,:,1]+im_blue[:,:,2])/(3.0*div))
	imageR = np.floor((im_red[:,:,0]+im_red[:,:,1]+im_red[:,:,2])/(3.0*div))
	imageG = np.floor((im_green[:,:,0]+im_green[:,:,1]+im_green[:,:,2])/(3.0*div))
	imageN = np.floor((im_nir[:,:,0]+im_nir[:,:,1]+im_nir[:,:,2])/(3.0*div))
	imageP = np.floor((im_pan[:,:,0]+im_pan[:,:,1]+im_pan[:,:,2])/(3.0*div))

	#split images into arrays
	splitB=utils.splitImage(imageB,N,num_blocks)
	splitR=utils.splitImage(imageR,N,num_blocks)
	splitG=utils.splitImage(imageG,N,num_blocks)
	splitN=utils.splitImage(imageN,N,num_blocks)
	splitP=utils.splitImage(imageP,N,num_blocks)

	meanB=block_mean(splitB)
	meanR=block_mean(splitR)
	meanG=block_mean(splitG)
	meanN=block_mean(splitN)
	meanP=block_mean(splitP)

	statsB=imStats(splitB,levels)
	statsR=imStats(splitR,levels)
	statsG=imStats(splitG,levels)
	statsN=imStats(splitN,levels)
	statsP=imStats(splitP,levels)

	height=heightFeatures.heightFeatures(im_height,N,"none")

	arrayImage=np.append(meanB,meanR,axis=0)
	arrayImage=np.append(arrayImage,meanG,axis=0)
	arrayImage=np.append(arrayImage,meanN,axis=0)
	arrayImage=np.append(arrayImage,meanP,axis=0)
	arrayImage=np.append(arrayImage,statsB,axis=0)
	arrayImage=np.append(arrayImage,statsR,axis=0)
	arrayImage=np.append(arrayImage,statsG,axis=0)
	arrayImage=np.append(arrayImage,statsN,axis=0)
	arrayImage=np.append(arrayImage,statsP,axis=0)
	arrayImage=np.append(arrayImage,height,axis=0)

	#Save data to file
	num_features=np.shape(arrayImage)[0]
	file_name="f"+str(num_features)+"_g"+str(levels)+"_b"+str(N)+"_"+filename

	np.save("../python features/"+file_name, arrayImage)

	print "Complete"


	
#ANDRA ENDAST DESSA VARDEN, STRANGAR OCH SOKVAGAR FIXAR SIG SJALVA
levels = 128 		#greyscale levels
N = 4 				#blockssize


filename="27"

featureExt(filename)
