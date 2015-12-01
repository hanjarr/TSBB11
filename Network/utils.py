import numpy as np 
import cv2
import os
import re
import glob
import math
import random
import scipy.ndimage.filters as sf
import matplotlib.pyplot as plt
from PIL import Image

np.set_printoptions(threshold=np.nan)

class Utils(object):

	def __init__(self, block_dim, num_features, num_classes, save_dir):
		self.block_dim =block_dim
		self.num_features = num_features
		self.num_classes = num_classes
		self.save_dir = save_dir

	def configureData(self, osm, features, training=False):

		'''Calculate number of blocks'''
		#dim=np.shape(osm)

		#block_dim = self.block_dim

		#num_blocks = dim**2/(block_dim**2)
		#print dim
		#print num_blocks
		num_blocks = np.shape(features)[1]

		'''Divide osm image into blocks '''
		#osm_blocks = splitImage(osm, block_dim, num_blocks)

		'''Load features for training and testing'''
		#feature_array = np.float32(np.load(features))
		feature_array = features

		'''Reshape osm blocks into arrays'''
		#osm_arrays = np.uint8(np.reshape(osm_blocks, (block_dim**2,num_blocks)))
		osm_arrays = osm

		inputs = [np.reshape(feature_array[:,y],(self.num_features,1)) for y in xrange(0,num_blocks)]
		if training:
			keys = [self.trainingLabel(osm_arrays[:,x]) for x in xrange(0,num_blocks)]
			training_data = zip(inputs,keys)
			input_data = self.reduceTrainingData(training_data)
		else:
			keys = [self.testLabel(osm_arrays[:,x]) for x in xrange(0,num_blocks)]
			input_data = zip(inputs,keys)
		return input_data

	def trainingLabel(self, label_array):
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


	def testLabel(self, label_array):
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
	def reduceTrainingData(self, training_data):

		random.shuffle(training_data)
		keys = [x[1] for x in training_data]

		num_keys = sum(keys)
		print num_keys
		min_key = np.min(num_keys)

		i=0
		k=0
		index=0
		reduced_data = []
		while (k<3):
			if(keys[index][k] == 1):
				reduced_data.append(training_data[index])
				i = i+1
				if i == min_key-1:
					i=0
					k = k+1
					index=0
			index +=1
		return reduced_data

		'''SORT OUT SAME AMOUNT OF EACH LABEL'''
	def duplicateTrainingData(self, training_data):

		keys = [x[1] for x in training_data]

		num_keys = sum(keys)
		max_key = np.max(num_keys)

		duplicated_data = []

		for i in range(0,len(num_keys)):
			separated_data = range(num_keys[i])
			duplication = int(math.ceil(max_key/num_keys[i]))

			index = 0

			for k in range(0,sum(num_keys)):
				if (keys[k][i] == 1):
					separated_data[index] = training_data[k]
					index += 1
			separated_data = separated_data*duplication
			duplicated_data += separated_data


		random.shuffle(duplicated_data)
		return duplicated_data
		
	def postProcessG(self,inImage):
		stl = np.shape(inImage)
		image = np.zeros(stl)
		image[:,:,0] = sf.gaussian_filter(inImage[:,:,0], 10)
		image[:,:,1] = sf.gaussian_filter(inImage[:,:,1], 5)
		image[:,:,2] = sf.gaussian_filter(inImage[:,:,2], 1.5)

		imSize = np.shape(image)

		for i in range(0,imSize[0]):
			for k in range(0,imSize[1]):
				maxValue = np.amax(image[i,k,:])
				for l in range(0,3):
					if(image[i,k,l] == maxValue):
						image[i,k,l] = 255
					else:
						image[i,k,l] = 0			
		
		image = Image.fromarray(image.astype(np.uint8))
		image.save(self.save_dir+'/processedG.png')
		return image

	def erodeDilate(self,im):
		
		im_size=np.shape(im)[0]

		image=np.zeros((im_size,im_size,3))

		kernel0 = np.ones((2,2),np.uint8)
		kernel1 = np.ones((5,5),np.uint8)
		kernel2 = np.ones((2,2),np.uint8)

		im0=im[:,:,0]
		im1=im[:,:,1]
		im2=im[:,:,2]

		im_op0= cv2.morphologyEx(im0,cv2.MORPH_OPEN,kernel0)
		im_op1 = cv2.morphologyEx(im1,cv2.MORPH_OPEN,kernel1)
		im_op2 = cv2.morphologyEx(im2,cv2.MORPH_OPEN,kernel2)
		
		image[:,:,0]=im_op0
		image[:,:,1]=im_op1
		image[:,:,2]=im_op2
		
		post=image
		for i in range(0,im_size):
			for j in range (0, im_size):
				if all(k==0 for k in image[i,j]):
					temp=np.zeros(3)
					si=i-1
					fi=i+1
					sj=j-1
					fj=j+1
					if i==0:
						si+=1
					if i==im_size-1:
						fi-=1
					if j==0:
						sj+=1
					if j==im_size-1:
						fj-=1
					for u in range(si,fi):
						for v in range(sj, fj):
							temp=temp+image[u,v]
					max_val=max(temp)
					temp_bool=(temp==max_val)
					new_color=(255*temp_bool*temp)/max_val
					post[i,j]=new_color
		image = Image.fromarray(post.astype(np.uint8))
		image.save(self.save_dir+"/erode_dilate.png")
		return image


	def createImage(self, network, test_data, test_osm, test_original):

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
		
		block_dim = self.block_dim
		test_image = np.multiply(np.array(resultImage[:,:,:]),255).astype(np.uint8)
		
		# PostProcess images with two different methods (use only one)
		test_image_processed = self.postProcessG(test_image)
		#test_image_processed = self.erodeDilate(test_image)
		
		# Convert from Image object to Numpy array
		test_image_array = np.array(test_image_processed.getdata(),np.uint8).reshape(test_image_processed.size[1], test_image_processed.size[0], 3)
		
		# Convert from BGR to RGB
		test_image_array = test_image_array[:,:,::-1].copy()
		
		# Scale up output image by block_dim
		image_resized = cv2.resize(test_image_array, (0,0), fx=block_dim, fy=block_dim,interpolation=0)
		cv2.imwrite(self.save_dir+"/out_RGB.png",image_resized)

		# Set all red pixels to white
		threshold = 100
		image_nored = image_resized
		image_nored[image_resized[:,:,2] > threshold,0] = 255
		image_nored[image_resized[:,:,2] > threshold,1] = 255
		image_nored[image_resized[:,:,2] > threshold,2] = 255
		cv2.imwrite(self.save_dir+"/out_nored.png",image_nored)

		# Blend result image with source image
		weight = 0.9 # Weight of source image in blended image, [0.0 - 1.0]
		image_source = cv2.imread(test_original)
		image_blended = cv2.addWeighted(image_resized,1-weight,image_source,weight,0)
		cv2.imwrite(self.save_dir+"/out_blended.png",image_blended)

		# Create error image
		#image_osm = cv2.imread(self.save_dir+"/test_osm.png")

		image_correct_red = np.logical_and(test_osm == 255,image_resized[:,:,2] == 255)
		image_correct_green = np.logical_and(test_osm == 119,image_resized[:,:,1] == 255)
		image_correct_blue = np.logical_and(test_osm == 0,image_resized[:,:,0] == 255)
		image_error = image_resized

		image_error[image_correct_red == True,0] = 255
		image_error[image_correct_red == True,1] = 255
		image_error[image_correct_red == True,2] = 255

		image_error[image_correct_green == True,0] = 255
		image_error[image_correct_green == True,1] = 255
		image_error[image_correct_green == True,2] = 255

		image_error[image_correct_blue == True,0] = 255
		image_error[image_correct_blue == True,1] = 255
		image_error[image_correct_blue == True,2] = 255

		cv2.imwrite(self.save_dir+"/out_error.png",image_error)
		test_image_processed.save(self.save_dir+"/output.png")

		return image_resized

	def evaluateResult(self, test_image, result_image):
		if(test_image.shape[0] != result_image.shape[0]):
			print "Error, not equal image sizes"
			return None
		else:
			accuracy, result = [], []
			labels = np.unique(test_image)
			
			layer = 0
			for label in labels:
				correct_pixels = 0
				result_pixels = 0
				for x in range(0,test_image.shape[0]):
					for y in range (0,test_image.shape[1]):
						if(result_image[x,y,layer] == 255 and test_image[x,y] == label):
							correct_pixels = correct_pixels+1
						if(test_image[x,y] == label):
							result_pixels +=1
				layer += 1
				accuracy.append(correct_pixels)
				result.append(result_pixels)
			
			percent = np.true_divide(accuracy,result)
			percent = percent[::-1].copy()
			
			print "Accuracy (Other, Water, Roads): ", 
			print percent
			
			return percent


	def plotAccuracy(self, accuracy, name):
		others = accuracy[0::self.num_classes]
		water = accuracy[1::self.num_classes]
		roads = accuracy[2::self.num_classes]

		line1, = plt.plot(others, 'r')
		line2, = plt.plot(water, 'g')
		line3, = plt.plot(roads, 'b')

		plt.ylabel('Accuracy')
		plt.xlabel('Epochs')

		plt.legend([line1,line2,line3],["Others", "Water", "Roads"])
		plt.savefig(self.save_dir + name + ".png")
		plt.clf()

	def plotCost(self, training_cost, evaluation_cost):
		line1, = plt.plot(training_cost)
		line2, = plt.plot(evaluation_cost)

		plt.ylabel('Cost')
		plt.xlabel('Epochs')
		plt.legend([line1,line2],["Training", "Evaluation"])
		plt.savefig(self.save_dir+"/cost.png")
		plt.clf()

	def plotConfusionMatrix(self, cm):
		labels=['Background', 'Water', 'Road']

		res = plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
		for i, cas in enumerate(cm):
			for j, c in enumerate(cas):
				if c>0:
					plt.text(j-.2, i+.2, "{0:.2f}".format(c), fontsize=14)

		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.xticks(range(self.num_classes), labels)
		plt.yticks(range(self.num_classes), labels)
		plt.colorbar()
		plt.savefig(self.save_dir + "/confusion")

	def loadData(self, training_osm, test_osm, training_features, test_features):

		training_data = self.configureData(training_osm, training_features, training = True)
		test_data = self.configureData(test_osm, test_features)

		return (training_data,test_data)

def splitImage(image_array, block_dim, num_blocks):

	image_blocks = np.zeros((block_dim,block_dim,num_blocks))

	iterations = int(num_blocks**0.5)

	i = 0
	for k in range(0,iterations):
		for j in range(0,iterations):
			image_blocks[:,:,i] = image_array[block_dim*k:block_dim+block_dim*k,block_dim*j:block_dim+block_dim*j]
			i += 1
	return image_blocks

def inputNetworkArray(osm_name, feature_name, im_numbers):
    total_feature_vector = []
    total_osm_vector = []
    j = 1
    for i in im_numbers:
        '''Load osm image'''
        osm = cv2.imread(str(osm_name+ str(i) + '.png'))
        if osm is None:
            raise IOError("cannot load file")
        osm = osm[:,:,0]
        '''Load features for training and testing'''
        feature_array = np.float32(np.load(str(feature_name + str(i) + '.npy')))

        '''Calculate number of blocks'''
        info = map(int, re.findall(r'\d+', feature_name))
    	block_dim = info[2]
        dim=np.shape(osm)[0]
        num_blocks = feature_array.shape[1] #lenght of feature vector

        '''Divide osm image into blocks '''
        osm_blocks = splitImage(osm,block_dim,num_blocks)
        '''Reshape osm blocks into arrays'''
        osm_arrays = np.uint8(np.reshape(osm_blocks, (block_dim**2,num_blocks)))
        '''Append vectors'''
        if j==1:
            total_osm_vector = osm_arrays
            total_feature_vector = feature_array
        else:
            total_osm_vector = np.append(total_osm_vector,osm_arrays,axis=1)
            total_feature_vector = np.append(total_feature_vector,feature_array,axis=1)
        j = j+1
    return total_osm_vector, total_feature_vector
