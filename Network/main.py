from network2 import Network
from utils import Utils, inputNetworkArray
import network2
import numpy as np
import random
import time
import json
import sys
import re
import os
import cv2

def main():


	'''Load osm images for training and test (validation)'''
	osm_path ="../images/divided images/rasterized"


	'''Choose features for training and test'''
	feature_path = "../python features/feature_vectors/f26_g128_b4_"

	im_numbers_train = [1,4,5,7,8,11,13,17,21,22,24,25,29,31,37,38,39,41,45,46,48,49,52,54,55,57,62]
	im_number_test = [18] #[2] [9] [16] [23] [30] 


	'''Load original for blending'''
	test_original = "../images/divided images/vricon_ortho_pan"+str(im_number_test[0])+".png"


	'''Use new func to generate array elements'''
	[training_osm_array, training_features_array] = inputNetworkArray(osm_path, feature_path, im_numbers_train)
	[test_osm_array, test_features_array] = inputNetworkArray(osm_path, feature_path, im_number_test)


	test_osm =cv2.imread(osm_path+str(im_number_test[0])+".png")[:,:,0]

	'''Choose features for training and test'''

	info = map(int, re.findall(r'\d+', feature_path))

	block_dim = info[2]
	input_layer = info[0]

	hidden_layer = 2*input_layer
	hidden_layer_2 = 20
	output_layer = 3

	epochs = 100
	mini_batch = 10
	learning_rate = 0.005

	sizes = [input_layer,hidden_layer, hidden_layer_2, output_layer]

	'''Create folder to save data in'''
	dirfmt = "../saved/%4d-%02d-%02d-%02d-%02d-%02d"
	save_dir = dirfmt % time.localtime()[0:6]
	os.mkdir(save_dir)

	'''Specify which net to load if you want to load an existing network'''
	load_net = "../saved/2015-12-04-13-30-13/network"

	utils = Utils(block_dim, input_layer, output_layer,training_osm_array, save_dir)
	training_data, test_data = utils.loadData(training_osm_array, test_osm_array, training_features_array, test_features_array)



	'''Load existing network from folder svaved_network'''
	if load_net:
		net = network2.load(load_net)
		total, accuracy, confusion = net.accuracy(test_data)
		data = {"best evaluation result": accuracy}

	else:
		net = Network(sizes)

		evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, test_confusion, data = \
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

		net = network2.load(save_dir+'/network')

	processed_result_resized, processed_stat_resized = utils.createImage(net, test_data, test_osm, test_original)
	processed_result_accuracy = utils.evaluateResult(test_osm,processed_result_resized).tolist()
	processed_stat_accuracy = utils.evaluateResult(test_osm,processed_stat_resized).tolist()

	data.update({"evaluation result after process G for original output image": processed_result_accuracy, "evaluation result after process G for statistical image": processed_stat_accuracy,
		"test image": im_number_test, "training_images": im_numbers_train})

	f = open(save_dir + "/info", "w")
	json.dump(data, f)
	f.close()

main()
