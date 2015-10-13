import cv2
import numpy as np
import scipy
import skimage
import scipy.ndimage.filters as sf
from skimage.feature import greycomatrix, greycoprops

np.set_printoptions(threshold = np.nan)

def imStats(image_block):
	pixel_pairs = greycomatrix(image_block, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4 ], levels=256, normed = False)
	pixel_pairs = np.array(pixel_pairs,dtype=np.float)
	measures = ['contrast','correlation','energy','homogeneity']
	
	numberOPP = [np.sum(pixel_pairs[:,:,:,0]), np.sum(pixel_pairs[:,:,:,1]), np.sum(pixel_pairs[:,:,:,2]), np.sum(pixel_pairs[:,:,:,3])]

	weightingOPP = np.zeros((256,256,1 ,1))
	weightingOPP[:,:,:,0] = np.divide(pixel_pairs[:,:,:,0],numberOPP[0]) + np.divide(pixel_pairs[:,:,:,1],numberOPP[1]) + np.divide(pixel_pairs[:,:,:,2],numberOPP[2]) + np.divide(pixel_pairs[:,:,:,3],numberOPP[3])


	#print np.unique(weightingOPP)
	feature_arrays = np.zeros((4))

	
	#print np.unique(pixel_pairs)

	for index,measure in enumerate(measures):
		feature_arrays[index] = greycoprops(weightingOPP,measure)
	
	return feature_arrays


# Import the images
im_blue = cv2.imread("../../sydney/ortho_blue/0_0_0_tex.tif")
im_red = cv2.imread("../../sydney/ortho_red/0_0_0_tex.tif")
im_green = cv2.imread("../../sydney/ortho_green/0_0_0_tex.tif")
im_nir = cv2.imread("../../sydney/ortho_nir/0_0_0_tex.tif")


# Create a RGB image 
redArray=np.array(im_red)
im = np.multiply(redArray, 0.2989)
greenSlice = im_green[:,:,1]
im[:,:,1] = np.multiply(np.array(greenSlice),0.5870)
blueSlice = im_blue[:,:,2]
im[:,:,2] = np.multiply(np.array(blueSlice),0.1140)

# Make the image to an uint8
#imUint8 = np.array(im,dtype=np.uint8)

# Divid the image
impart = im[4000:4500, 2000:2500, :]

# Filtrating the image
inImage = impart
inImage[:,:,0] = sf.gaussian_filter(impart[:,:,0], 8.0)
inImage[:,:,1] = sf.gaussian_filter(impart[:,:,1], 8.0)
inImage[:,:,2] = sf.gaussian_filter(impart[:,:,2], 8.0)

totalImage = np.divide(inImage[:,:,0]+inImage[:,:,1]+inImage[:,:,2],3.0)

# Variables 
inImSize = np.shape(totalImage)
N = 4
upperLimits = np.array([np.floor(inImSize[0]/N)*N, np.floor(inImSize[1]/N)*N])
image = totalImage[1:upperLimits[0],1:upperLimits[1]]
# imSize = size(image);

# An empty image
newImage = np.zeros((upperLimits[0]/N-1, upperLimits[1]/N-1, 4))
arrayImage = np.zeros((4,(upperLimits[1]-N)/N+((upperLimits[0]-N)*(upperLimits[1]-N))/N))

# Creating features contrast, energy, correlation, homogeneity
for i in range(0,int(upperLimits[0]-N), N):
	for k in range (0,int(upperLimits[1]-N), N):
		extImage = image[i:i+N, k:k+N]
		stats = imStats(extImage)
		#print k
		
		newImage[(i/N), (k/N), 0] = stats[0];
		newImage[(i/N), (k/N), 1] = stats[1];
		newImage[(i/N), (k/N), 2] = stats[2];
		newImage[(i/N), (k/N), 3] = stats[3];
		
		#Reshape to array
		arrayImage[:,k/N+(i*(upperLimits[1]-N)/N)] = stats
		

	print i/(upperLimits[0]-N)


#print newImage[:,:,0]
print np.shape(arrayImage)
print type(arrayImage)

contrast = np.multiply(np.divide(np.array(newImage[:,:,0],dtype=np.uint8), np.amax(newImage[:,:,0])),255)
correlation = np.multiply(np.divide(np.array(newImage[:,:,1],dtype=np.uint8), np.amax(newImage[:,:,1])),255)
energy = np.multiply(np.divide(np.array(newImage[:,:,2],dtype=np.uint8), np.amax(newImage[:,:,2])),255)
homogeneity = np.multiply(np.divide(np.array(newImage[:,:,3],dtype=np.uint8), np.amax(newImage[:,:,3])),255)

arrayImage.tofile("features.ext")

cv2.imshow("contrast", contrast)
cv2.imshow("correlation", correlation)
cv2.imshow("energy", energy)
cv2.imshow("homogeneity", homogeneity)
cv2.waitKey(0)





print "Hello world"


