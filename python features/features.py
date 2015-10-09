import cv2
import numpy as np
import scipy
import skimage
import scipy.ndimage.filters as sf

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

# Variables 
inImSize = np.shape(inImage)
N = 4
upperLimits = np.array([np.floor(inImSize[0]/N)*N, np.floor(inImSize[1]/N)*N])
image = inImage[1:upperLimits[0],1:upperLimits[1],:]
# imSize = size(image);

# An empty image
newImage = np.zeros(upperLimits[0]/N-1, upperLimits[1]/N-1, 4);

# Creating features contrast, energy, correlation, homogeneity
for i in xrange (0,upperLimits[0]-N, N)
	for k in xrange (0, upperLimits[1]-N, N)
		extImage = image[i:i+(N-1), k:k+(N-1)]
        stats = imStats(extImage)
        
        newImage[((i-1)/N)+1, ((k-1)/N)+1, 1] = stats.Contrast;
        newImage[((i-1)/N)+1, ((k-1)/N)+1, 2] = stats.Correlation;
        newImage[((i-1)/N)+1, ((k-1)/N)+1, 3] = stats.Energy;
        newImage[((i-1)/N)+1, ((k-1)/N)+1, 4] = stats.Homogeneity;

    print i/(upperLimits[0]-N)

# Normalization
newImage[:,:,1] = newImage[:,:,1]/np.amax(newImage[:,:,1]);
newImage[:,:,2] = newImage[:,:,2]/np.amax(newImage[:,:,2]);
newImage[:,:,3] = newImage[:,:,3]/np.amax(newImage[:,:,3]);
newImage[:,:,4] = newImage[:,:,4]/np.amax(newImage[:,:,4]);

print "Hello world"


