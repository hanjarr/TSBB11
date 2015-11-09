import cv2
import numpy as np
import scipy
import skimage
import scipy.ndimage.filters as sf
from skimage.feature import greycomatrix, greycoprops
from PIL import Image

np.set_printoptions(threshold = np.nan)

def imStats(image_block,levels):
	pixel_pairs = greycomatrix(image_block, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4 ], levels, normed = False)
	pixel_pairs = np.array(pixel_pairs,dtype=np.float)
	measures = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
	
	numberOPP = [np.sum(pixel_pairs[:,:,:,0]), np.sum(pixel_pairs[:,:,:,1]), np.sum(pixel_pairs[:,:,:,2]), np.sum(pixel_pairs[:,:,:,3])]

	weightingOPP = np.zeros((levels,levels,1 ,1))
	weightingOPP[:,:,:,0] = np.divide(pixel_pairs[:,:,:,0],numberOPP[0]) + np.divide(pixel_pairs[:,:,:,1],numberOPP[1]) + np.divide(pixel_pairs[:,:,:,2],numberOPP[2]) + np.divide(pixel_pairs[:,:,:,3],numberOPP[3])
	
	feature_arrays = np.zeros((num_imfeatures))

	for index,measure in enumerate(measures):
		feature_arrays[index] = greycoprops(weightingOPP,measure)
	
	feature_arrays[num_imfeatures-1]=image_block.mean()
	return feature_arrays



def gaussFilt(im,filt_const):
	image = im
	image[:,:,0] = sf.gaussian_filter(im[:,:,0], filt_const)
	image[:,:,1] = sf.gaussian_filter(im[:,:,1], filt_const)
	image[:,:,2] = sf.gaussian_filter(im[:,:,2], filt_const)

	return image



levels = 16 #greyscale levels
N = 4 #blockssize
num_im=4
gaussNr=16
file_name="f16_g16_b4_gau16_Test"

global num_imfeatures
num_imfeatures=7


#constants
num_features=num_im*num_imfeatures
div=256/levels


# Import the images
im_blue = cv2.imread("../images/test_blue.png")
im_red = cv2.imread("../images/test_red.png")
im_green = cv2.imread("../images/test_green.png")
im_nir = cv2.imread("../images/test_nir.png")
#im_pan=cv2.imread("../images/train_pan.png")

# Filtrating the image
inImageR = gaussFilt(im_red,gaussNr)
inImageG = gaussFilt(im_green,gaussNr)
inImageB = gaussFilt(im_blue,gaussNr)
inImageN = gaussFilt(im_nir,gaussNr)
#inImageP = gaussFilt(im_pan,8)

imGaussRed = Image.fromarray(inImageR)
imGaussRed.save('gaussRed16Test.png')

imGaussGreen = Image.fromarray(inImageG)
imGaussGreen.save('gaussGreen16Test.png')

imGaussBlue = Image.fromarray(inImageB)
imGaussBlue.save('gaussBlue16Test.png')

imGaussNir = Image.fromarray(inImageN)
imGaussNir.save('gaussNir16Test.png')

#
totalImageR = np.divide(inImageR[:,:,0]+inImageR[:,:,1]+inImageR[:,:,2],3.0)
totalImageG = np.divide(inImageG[:,:,0]+inImageG[:,:,1]+inImageG[:,:,2],3.0)
totalImageB = np.divide(inImageB[:,:,0]+inImageB[:,:,1]+inImageB[:,:,2],3.0)
totalImageN = np.divide(inImageN[:,:,0]+inImageN[:,:,1]+inImageN[:,:,2],3.0)
#totalImageP = np.divide(inImageP[:,:,0]+inImageP[:,:,1]+inImageP[:,:,2],3.0)

# Variables 
inImSize = np.shape(totalImageR)

upperLimits = np.array([np.floor(inImSize[0]/N)*N, np.floor(inImSize[1]/N)*N])

imageR = totalImageR[1:upperLimits[0],1:upperLimits[1]]/div
imageG = totalImageG[1:upperLimits[0],1:upperLimits[1]]/div
imageB = totalImageB[1:upperLimits[0],1:upperLimits[1]]/div
imageN = totalImageN[1:upperLimits[0],1:upperLimits[1]]/div
#imageP = totalImageP[1:upperLimits[0],1:upperLimits[1]]/div
# An empty image
arrayImage = np.zeros((num_features,(upperLimits[1]/N)*(upperLimits[0]/N)))

# Creating features 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', mean
for i in range(0,int(upperLimits[0]-N), N):
	for k in range (0,int(upperLimits[1]-N), N):
		
		extImage = imageR[i:i+N, k:k+N]
		statsR = imStats(extImage,levels)
		
		extImage = imageG[i:i+N, k:k+N]
		statsG = imStats(extImage,levels)
		
		extImage = imageB[i:i+N, k:k+N]
		statsB = imStats(extImage,levels)
		
		extImage = imageN[i:i+N, k:k+N]
		statsN = imStats(extImage,levels)
		
		#extImage = imageP[i:i+N, k:k+N]
		#statsP = imStats(extImage,levels)

		statsTemp = np.append(statsR, statsG)
		statsTemp2 = np.append(statsB, statsN)
		
		stats = np.append(statsTemp, statsTemp2)
		
		#Reshape to array
		arrayImage[:,k/N+(i*(upperLimits[1])/N**2)] = stats
		
		
	print i/(upperLimits[0]-N)

#normalize data 
for j in range(0,num_features-1):
	arrayImage[j,:]= np.divide(arrayImage[j,:],np.amax(arrayImage[j,:]))


#Save data to file
np.save(file_name, arrayImage)

print "GRATTIS <3"


