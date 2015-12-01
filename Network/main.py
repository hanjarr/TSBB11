from network2 import Network 
from utils import Utils, inputNetworkArray
import numpy as np
import random
import time
import re
import os
import cv2

def main():

	
	'''Load osm images for training and test (validation)'''
	training_osm_str = "../images/divided images/rasterized"
	test_osm_str ="../images/divided images/rasterized"


	'''Choose features for training and test'''
	training_features = "../python features/f6_g64_b4_gau4_"
	test_features = "../python features/f6_g64_b4_gau4_"

	im_numbers_train = [1,3]
	im_numbers_test = [2]

	'''Load original for blending'''
	test_original = "../images/divided images/vricon_ortho_pan"+str(im_numbers_test[0])+".png"


	'''Use new func to generate array elements'''
	[training_osm_array, training_features_array] = inputNetworkArray(training_osm_str, training_features, im_numbers_train)
	[test_osm_array, test_features_array] = inputNetworkArray(test_osm_str, test_features, im_numbers_test)


	#training_osm = cv2.imread(training_osm_str)[:,:,0]
	test_osm =cv2.imread(test_osm_str+str(im_numbers_test[0])+".png")[:,:,0]

	'''Choose features for training and test'''

	#training_features = "../python features/graylevels/f35_g128_b4_gau4_train.npy"
	#test_features = "../python features/graylevels/f35_g128_b4_gau4_test.npy"

	info = map(int, re.findall(r'\d+', test_features))

	block_dim = info[2]
	input_layer = info[0]

	hidden_layer = 2*input_layer
	hidden_layer_2 = 20
	output_layer = 3

	epochs = 2
	mini_batch = 10
	learning_rate = 0.008

	'''Create folder to save data in'''
	dirfmt = "../saved/%4d-%02d-%02d-%02d-%02d-%02d"
	save_dir = dirfmt % time.localtime()[0:6]
	os.mkdir(save_dir)

	'''Specify which net to load if you want to load an existing network'''
	load_net = ""#../saved_networks/e15_mb10_lr0.01f35_g128_b4_gau4_train.npy"

	utils = Utils(block_dim, input_layer, output_layer, save_dir)
	training_data, test_data = utils.loadData(training_osm_array, test_osm_array, training_features_array, test_features_array)



	'''Load existing network from folder svaved_network'''
	if load_net:
		net = network2.load(load_net)
		net.accuracy(test_data)
	else:
		net = Network([input_layer,hidden_layer, output_layer])

		evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, test_confusion = \
		net.SGD(training_data, epochs, mini_batch, learning_rate, save_dir,lmbda=0.5, evaluation_data = test_data, \
		monitor_training_cost = True, monitor_evaluation_cost = True, monitor_training_accuracy=True, monitor_evaluation_accuracy=True)

		''' Create accuracy plot'''
		if evaluation_accuracy:
			utils.plotAccuracy(evaluation_accuracy, "/evaluation_accuracy")

		if training_accuracy:
			utils.plotAccuracy(training_accuracy, "/training_accuracy")

		''' Create cost plot'''
		if evaluation_cost and training_cost:
			utils.plotCost(training_cost,evaluation_cost)

		'''Create confusion matrix'''
		if test_confusion.any():
			utils.plotConfusionMatrix(test_confusion)

	image_resized = utils.createImage(net, test_data, test_osm, test_original)
	utils.evaluateResult(test_osm,image_resized)	
main()