"""main.py
~~~~~~~~~~~~~~

A module to run the system. Includes initialization of network parameters, 
choise of training and test images, etc. 

"""

from network import Network
from utils import Utils, inputNetworkArray
import network
import numpy as np
import random
import time
import json
import sys
import re
import os
import cv2

def main():

	''' Specify path to images'''
	osm_path ="../images/rasterized"


	''' Specify path to features'''
	feature_path = "../features/f26_g128_b4_"

	''' Choose cutouts to use for training and testing'''
	im_numbers_train = [1,4,5,7,8,11,13,17,21,22,24,25,29,31,37,38,39,41,45,46,48,49,52,54,55,57,62]
	im_number_test = [18]


	''' Load original image for blending'''
	test_original = "../images/vricon_ortho_pan"+str(im_number_test[0])+".png"


	''' Generate array elements'''
	[test_osm_array, test_features_array] = inputNetworkArray(osm_path, feature_path, im_number_test)
	[training_osm_array, training_features_array] = inputNetworkArray(osm_path, feature_path, im_numbers_train)


	''' Create folder to save data in'''
	dirfmt = "../saved/%4d-%02d-%02d-%02d-%02d-%02d"
	save_dir = dirfmt % time.localtime()[0:6]
	os.mkdir(save_dir)

	''' Specify which net to load if you want to load an existing network. Specify empty string "" the training 
	module is wanted '''
	load_net = "../saved/2015-12-09-18-05-21/network"

	''' Imread ground truth test image''' 
	test_osm =cv2.imread(osm_path+str(im_number_test[0])+".png")[:,:,0]

	''' Choose features for training and test'''
	info = map(int, re.findall(r'\d+', feature_path))
	
	''' Specify network parameters'''
	block_dim = info[2]
	input_layer = info[0]

	hidden_layer = 2*input_layer
	hidden_layer_2 = 20
	output_layer = 3

	epochs = 10
	mini_batch = 10
	learning_rate = 0.005

	sizes = [input_layer,hidden_layer, hidden_layer_2, output_layer]

	'''Create utils object'''
	utils = Utils(block_dim, input_layer, output_layer,training_osm_array, save_dir)

	''' Configure validation data'''
	test_data = utils.loadData(test_osm_array, test_features_array)


	''' Load existing network from folder saved_network if it is specified'''
	if load_net:
		net = network.load(load_net)
		total, accuracy, confusion = net.accuracy(test_data)
		data = {"best evaluation result": accuracy}

	else:
		net = Network(sizes)

		''' Configure training data'''
		training_data = utils.loadData(training_osm_array, training_features_array, training = True)

		''' Train the network using SGD'''
		evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, test_confusion, data = \
		net.SGD(training_data, epochs, mini_batch, learning_rate, save_dir,lmbda=0.5, evaluation_data = test_data, \
		monitor_training_cost = True, monitor_evaluation_cost = True, monitor_training_accuracy=True, monitor_evaluation_accuracy=True)

		''' Create accuracy plots'''
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

		''' Load the best network after training for testing'''
		net = network.load(save_dir+'/network')


	''' Create images from the classifications'''
	processed_result_resized, processed_stat_resized = utils.createImage(net, test_data, test_osm, test_original)

	''' Calculate accuracy after processing '''
	processed_result_accuracy = utils.evaluateResult(test_osm, processed_result_resized).tolist()
	processed_stat_accuracy = utils.evaluateResult(test_osm, processed_stat_resized).tolist()

	''' Update info file that is generated'''
	data.update({"evaluation result after process G for original output image": processed_result_accuracy, 
		"evaluation result after process G for statistical image": processed_stat_accuracy,
		"test image": im_number_test, "training_images": im_numbers_train})

	''' Saves info file to specified folder'''
	f = open(save_dir + "/info", "w")
	json.dump(data, f)
	f.close()

main()
