import numpy as np 
import cv2
import os
import glob
from PIL import Image

np.set_printoptions(threshold=np.nan)

class Utils(object):

    def __init__(self, block_dim, num_features, num_classes):
    	self.block_dim =block_dim
    	self.num_features = num_features
    	self.num_classes = num_classes

    def split_image(self, image_array, num_blocks):

    	block_dim = self.block_dim
    	image_blocks = np.zeros((block_dim,block_dim,num_blocks))

    	iterations = int(num_blocks**0.5)

    	i = 0
    	for k in range(0,iterations):
    		for j in range(0,iterations):
    			image_blocks[:,:,i] = image_array[block_dim*k:block_dim+block_dim*k,block_dim*j:block_dim+block_dim*j]
    			i += 1
    	return image_blocks

    def configureData(self, osm, features, training=False):

    	'''Calculate number of blocks'''
    	dim=np.shape(osm)[0]
    	num_blocks = dim**2/(self.block_dim**2)

    	'''Divide osm image into blocks '''
    	osm_blocks = self.split_image(osm, num_blocks)


    	'''Load features for training and testing'''
    	feature_array = np.float32(np.load(features))

    	'''Reshape osm blocks into arrays'''
    	osm_arrays = np.uint8(np.reshape(osm_blocks, (self.block_dim**2,num_blocks)))

    	inputs = [np.reshape(feature_array[:,y],(self.num_features,1)) for y in xrange(0,num_blocks)]
    	if training:
    		keys = [self.training_label(osm_arrays[:,x]) for x in xrange(0,num_blocks)]
    		training_data = zip(inputs,keys)
    		input_data = self.reduceTrainingData(keys, training_data)
    	else:
    		keys = [self.test_label(osm_arrays[:,x]) for x in xrange(0,num_blocks)]
    		input_data = zip(inputs,keys)
    	return input_data

    def training_label(self, label_array):
    	class_array = np.zeros((self.num_classes,1))
    	counts = np.bincount(label_array)
    	block_class = np.argmax(counts)
    	if block_class == 255:
    		class_array[0]=1.0
    	elif block_class == 119:
    		class_array[1]=1.0
    	else:
    		class_array[2]=1.0
    	return class_array


    def test_label(self, label_array):
    	counts = np.bincount(label_array)
    	block_class = np.argmax(counts)
    	if block_class == 255:
    		class_label=0
    	elif block_class == 119:
    		class_label=1
    	else:
    		class_label=2
    	return np.int64(class_label)


    '''SORT OUT SAME AMOUNT OF EACH LABEL'''
    def reduceTrainingData(self, keys, training_data):
    	num_keys = sum(keys)
    	min_key = np.min(num_keys)

    	i=0
    	k=0
    	index=0
    	new_index = 0
    	reduced_data = [training_data[x] for x in xrange(0,int(3*min_key))]
    	while (k<3):
    		if(keys[index][k] == 1):
    			reduced_data[new_index] = training_data[index]
    			new_index = new_index+1
    			i = i+1
    			if i == min_key:
    				i=0
    				k = k+1
    				index=0
    		index=index+1
    	return reduced_data

    def createImage(self, network, test_data):

		test_results = [(np.argmax(network.feedforward(x)), y)	for (x, y) in test_data]

		resultDim = len(test_results)
		imDim = np.sqrt(resultDim)
		resultImage = np.zeros((int(imDim),int(imDim),3))
		u=0
		v=0
		for i in range(0,int(resultDim)):
			resultImage[v,u,test_results[i][0]] = 1
			u=u+1
			if(u==imDim):
				u=0
				v=v+1
		test_image = np.multiply(np.array(resultImage[:,:,:]),255).astype(np.uint8)
		
		im = Image.fromarray(test_image)
		im.save('f16_g16_b4_gau16.png')

	    #cv2.imshow("RESULTAT", test_image)
	    #cv2.waitKey(0)

    def load_data(self):

    	'''Load osm images for training and test (validation)'''
    	training_osm = cv2.imread("../images/train_osm.png")[:,:,0]
    	test_osm =cv2.imread("../images/test_osm.png")[:,:,0]

    	training_data = self.configureData(training_osm, "../python features/SavedImages/f16_g16_b4_gau16_train.npy", training=True)
    	test_data = self.configureData(test_osm, "../python features/SavedImages/f16_g16_b4_gau16_Test.npy")

    	return (training_data,test_data)
