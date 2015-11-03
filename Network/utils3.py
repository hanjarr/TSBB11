import numpy as np 
import cv2
import os
import glob
from skimage.feature import greycomatrix, greycoprops

def split_image(image_array, block_dim, vricon_data = False):

	image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

	image_dim = np.shape(image_array);
	num_blocks = image_dim[0]**2/(block_dim**2)
	image_blocks = np.zeros((block_dim,block_dim,num_blocks))

	iterations = image_dim[0]/block_dim

	i = 0
	for k in range(0,iterations):
	    for j in range(0,iterations):
	    	image_blocks[:,:,i] = image_array[block_dim*k:block_dim+block_dim*k,block_dim*j:block_dim+block_dim*j]
	    	i += 1
	return image_blocks


def load_data(block_dim):

	np.set_printoptions(threshold=np.nan)

	osm = cv2.imread("osm.png")
	osm_blocks = split_image(osm, block_dim)

	image_dim = np.shape(osm);
	num_blocks = image_dim[0]**2/(block_dim**2)

	training_blocks = int(num_blocks*0.8)
	test_blocks = num_blocks - training_blocks

	features = "../python features/features.npy"
	vricon_arrays = np.load(features)

	osm_arrays = np.uint8(np.reshape(osm_blocks, (block_dim**2,num_blocks)))

	tr_inputs = np.float32(vricon_arrays[:,0:training_blocks])
	tr_keys = osm_arrays[:,0:training_blocks]

	te_inputs = np.float32(vricon_arrays[:,training_blocks:])
	te_keys = osm_arrays[:,training_blocks:]

	training_inputs = [np.reshape(tr_inputs[:,y],(4,1)) for y in xrange(0,training_blocks)]
	training_keys = [training_label(tr_keys[:,x]) for x in xrange(0,training_blocks)]
	training_data = zip(training_inputs,training_keys)
	
	#SORT OUT SAME AMOUNT OF EACH LABLE
	sumOfAll = sum(training_keys)
	numOfEach = np.min(sumOfAll)
	
	i=0
	k=0
	index=0
	newIndex = 0
	newTrainingData = [training_data[x] for x in xrange(0,int(3*numOfEach))]
	
	while (k<3):
		
		if(training_keys[index][k] == 1):
			newTrainingData[newIndex] = training_data[index]
			newIndex = newIndex+1
			i = i+1
			
			if i == numOfEach:
				i=0
				k = k+1
				index=0
			
		index=index+1
	
	training_data = newTrainingData
	#END OF SORT
	
	test_inputs = [np.reshape(te_inputs[:,y],(4,1)) for y in xrange(0,test_blocks)]
	test_keys= [test_label(te_keys[:,x]) for x in xrange(0,test_blocks)]
	test_data = zip(test_inputs,np.int64(test_keys))

	label_weights = sum(training_keys)/(sum(training_keys)[0])
	return (training_data,test_data,label_weights)


def training_label(label_array):
	class_array = np.zeros((3,1))
	counts = np.bincount(label_array)
	block_class = np.argmax(counts)
	if block_class == 255:
		class_array[0]=1.0
	elif block_class == 232:
		class_array[1]=1.0
	else:
		class_array[2]=1.0
	return class_array


def test_label(label_array):
	counts = np.bincount(label_array)
	block_class = np.argmax(counts)
	if block_class == 255:
		class_label=0
	elif block_class == 232:
		class_label=1
	else:
		class_label=2
	return class_label

	
def createImage(network, block_dim):
	#Reload Data
	osm = cv2.imread("osm.png")
	osm_blocks = split_image(osm, block_dim)

	image_dim = np.shape(osm);
	num_blocks = image_dim[0]**2/(block_dim**2)

	training_blocks = int(num_blocks)
	test_blocks = num_blocks - training_blocks

	features = "../python features/features.npy"
	vricon_arrays = np.load(features)

	osm_arrays = np.uint8(np.reshape(osm_blocks, (block_dim**2,num_blocks)))

	tr_inputs = np.float32(vricon_arrays[:,0:training_blocks])
	tr_keys = osm_arrays[:,0:training_blocks]

	te_inputs = np.float32(vricon_arrays[:,training_blocks:])
	te_keys = osm_arrays[:,training_blocks:]

	training_inputs = [np.reshape(tr_inputs[:,y],(4,1)) for y in xrange(0,training_blocks)]
	training_keys = [training_label(tr_keys[:,x]) for x in xrange(0,training_blocks)]
	training_data = zip(training_inputs,training_keys)
	#End Reload Data
	
	test_results = [(np.argmax(network.feedforward(x)), y)	for (x, y) in training_data]
	
	resultDim = len(test_results)
	imDim = np.sqrt(resultDim)
	print imDim
	resultImage = np.zeros((int(imDim),int(imDim),3))
	print test_results[100][0]
	u=0
	v=0
	for i in range(0,int(resultDim)):
		resultImage[v,u,test_results[i][0]] = 1
		u=u+1
		if(u==imDim):
			u=0
			v=v+1
	
	resultImage = np.multiply(np.array(resultImage[:,:,:]),255)
	
	cv2.imshow("RESULTAT", resultImage)
	cv2.waitKey(0)
		