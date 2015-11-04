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
im_blue = cv2.imread("../images/ortho_blue.png")
im_red = cv2.imread("../images/ortho_red.png")
im_green = cv2.imread("../images/ortho_green.png")
im_nir = cv2.imread("../../sydney/ortho_nir/0_0_0_tex.tif")


# Create a RGB image 
#redArray=np.array(im_red)
#im = np.multiply(redArray, 0.2989)
#greenSlice = im_green[:,:,1]
#im[:,:,1] = np.multiply(np.array(greenSlice),0.5870)
#blueSlice = im_blue[:,:,2]
#im[:,:,2] = np.multiply(np.array(blueSlice),0.1140)

# Make the image to an uint8
#imUint8 = np.array(im,dtype=np.uint8)

# Divid the image
#impart = im

# Filtrating the image
inImageR = im_red
inImageR[:,:,0] = sf.gaussian_filter(im_red[:,:,0], 8.0)
inImageR[:,:,1] = sf.gaussian_filter(im_red[:,:,1], 8.0)
inImageR[:,:,2] = sf.gaussian_filter(im_red[:,:,2], 8.0)

inImageG = im_green
inImageG[:,:,0] = sf.gaussian_filter(im_green[:,:,0], 8.0)
inImageG[:,:,1] = sf.gaussian_filter(im_green[:,:,1], 8.0)
inImageG[:,:,2] = sf.gaussian_filter(im_green[:,:,2], 8.0)

inImageB = im_blue
inImageB[:,:,0] = sf.gaussian_filter(im_blue[:,:,0], 8.0)
inImageB[:,:,1] = sf.gaussian_filter(im_blue[:,:,1], 8.0)
inImageB[:,:,2] = sf.gaussian_filter(im_blue[:,:,2], 8.0)

inImageN = im_nir
inImageN[:,:,0] = sf.gaussian_filter(im_nir[:,:,0], 8.0)
inImageN[:,:,1] = sf.gaussian_filter(im_nir[:,:,1], 8.0)
inImageN[:,:,2] = sf.gaussian_filter(im_nir[:,:,2], 8.0)

totalImageR = np.divide(inImageR[:,:,0]+inImageR[:,:,1]+inImageR[:,:,2],3.0)
totalImageG = np.divide(inImageG[:,:,0]+inImageG[:,:,1]+inImageG[:,:,2],3.0)
totalImageB = np.divide(inImageB[:,:,0]+inImageB[:,:,1]+inImageB[:,:,2],3.0)
totalImageN = np.divide(inImageN[:,:,0]+inImageN[:,:,1]+inImageN[:,:,2],3.0)

# Variables 
inImSize = np.shape(totalImageR)
N = 4
upperLimits = np.array([np.floor(inImSize[0]/N)*N, np.floor(inImSize[1]/N)*N])

imageR = totalImageR[1:upperLimits[0],1:upperLimits[1]]
imageG = totalImageG[1:upperLimits[0],1:upperLimits[1]]
imageB = totalImageB[1:upperLimits[0],1:upperLimits[1]]
imageN = totalImageN[1:upperLimits[0],1:upperLimits[1]]

# An empty image
#newImage = np.zeros((upperLimits[0]/N, upperLimits[1]/N, 4))
arrayImage = np.zeros((16,(upperLimits[1]/N)*(upperLimits[0]/N)))

# Creating features contrast, energy, correlation, homogeneity
for i in range(0,int(upperLimits[0]-N), N):
	for k in range (0,int(upperLimits[1]-N), N):
		
		extImage = imageR[i:i+N, k:k+N]
		statsR = imStats(extImage)
		
		extImage = imageG[i:i+N, k:k+N]
		statsG = imStats(extImage)
		
		extImage = imageB[i:i+N, k:k+N]
		statsB = imStats(extImage)
		
		extImage = imageN[i:i+N, k:k+N]
		statsN = imStats(extImage)
		
		statsTemp = np.append(statsR, statsG)
		statsTemp2 = np.append(statsB, statsN)
		
		stats = np.append(statsTemp, statsTemp2)
		
		#Reshape to array
		arrayImage[:,k/N+(i*(upperLimits[1])/N**2)] = stats
		
		#newImage[(i/N), (k/N), 0] = stats[0];
		#newImage[(i/N), (k/N), 1] = stats[1];
		#newImage[(i/N), (k/N), 2] = stats[2];
		#newImage[(i/N), (k/N), 3] = stats[3];

	print i/(upperLimits[0]-N)


#print newImage[:,:,0]
print np.shape(arrayImage)
print type(arrayImage)

#Normalization and converting to uint8
#contrast = np.multiply(np.divide(np.array(newImage[:,:,0],dtype=np.uint8), np.amax(newImage[:,:,0])),255)
#correlation = np.multiply(np.divide(np.array(newImage[:,:,1],dtype=np.uint8), np.amax(newImage[:,:,1])),255)
#energy = np.multiply(np.divide(np.array(newImage[:,:,2],dtype=np.uint8), np.amax(newImage[:,:,2])),255)
#homogeneity = np.multiply(np.divide(np.array(newImage[:,:,3],dtype=np.uint8), np.amax(newImage[:,:,3])),255)

arrayImage[0,:]= np.divide(arrayImage[0,:],np.amax(arrayImage[0,:]))
arrayImage[1,:]= np.divide(arrayImage[1,:],np.amax(arrayImage[1,:]))
arrayImage[2,:]= np.divide(arrayImage[2,:],np.amax(arrayImage[2,:]))
arrayImage[3,:]= np.divide(arrayImage[3,:],np.amax(arrayImage[3,:]))
arrayImage[4,:]= np.divide(arrayImage[4,:],np.amax(arrayImage[4,:]))
arrayImage[5,:]= np.divide(arrayImage[5,:],np.amax(arrayImage[5,:]))
arrayImage[6,:]= np.divide(arrayImage[6,:],np.amax(arrayImage[6,:]))
arrayImage[7,:]= np.divide(arrayImage[7,:],np.amax(arrayImage[7,:]))
arrayImage[8,:]= np.divide(arrayImage[8,:],np.amax(arrayImage[8,:]))
arrayImage[9,:]= np.divide(arrayImage[9,:],np.amax(arrayImage[9,:]))
arrayImage[10,:]= np.divide(arrayImage[10,:],np.amax(arrayImage[10,:]))
arrayImage[11,:]= np.divide(arrayImage[11,:],np.amax(arrayImage[11,:]))
arrayImage[12,:]= np.divide(arrayImage[12,:],np.amax(arrayImage[12,:]))
arrayImage[13,:]= np.divide(arrayImage[13,:],np.amax(arrayImage[13,:]))
arrayImage[14,:]= np.divide(arrayImage[14,:],np.amax(arrayImage[14,:]))
arrayImage[15,:]= np.divide(arrayImage[15,:],np.amax(arrayImage[15,:]))

#Save data to file
np.save("features", arrayImage)

#Display features
#cv2.imshow("contrast", contrast)
#cv2.imshow("correlation", correlation)
#cv2.imshow("energy", energy)
#cv2.imshow("homogeneity", homogeneity)
#cv2.waitKey(0)





print "Hello world"


