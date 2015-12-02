from network2 import Network 
from utils import Utils, inputNetworkArray
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
	osm_str ="../images/divided images/rasterized"


	'''Choose features for training and test'''
	training_features = "../python features/f26_g128_b4_"
	test_features = "../python features/f26_g128_b4_"

	im_numbers_train = [1,4,5,10,17,25,27,31,39,41,42,43,48,49,52,55,57]
	im_numbers_test = [18]

	'''Load original for blending'''
	test_original = "../images/divided images/vricon_ortho_pan"+str(im_numbers_test[0])+".png"


	'''Use new func to generate array elements'''
	[training_osm_array, training_features_array] = inputNetworkArray(osm_str, training_features, im_numbers_train)
	[test_osm_array, test_features_array] = inputNetworkArray(osm_str, test_features, im_numbers_test)


	test_osm =cv2.imread(osm_str+str(im_numbers_test[0])+".png")[:,:,0]

	'''Choose features for training and test'''

	info = map(int, re.findall(r'\d+', test_features))

	block_dim = info[2]
	input_layer = info[0]

	hidden_layer = 2*input_layer
	hidden_layer_2 = 20
	output_layer = 3

	epochs = 1
	mini_batch = 10
	learning_rate = 0.008

	sizes = [input_layer,hidden_layer, hidden_layer_2, output_layer]

	'''Create folder to save data in'''
	dirfmt = "../saved/%4d-%02d-%02d-%02d-%02d-%02d"
	save_dir = dirfmt % time.localtime()[0:6]
	os.mkdir(save_dir)

	'''Specify which net to load if you want to load an existing network'''
	load_net = ""

	utils = Utils(block_dim, input_layer, output_layer, save_dir)
	training_data, test_data = utils.loadData(training_osm_array, test_osm_array, training_features_array, test_features_array)



	'''Load existing network from folder svaved_network'''
	if load_net:
		net = network2.load(load_net)
		net.accuracy(test_data)
	else:
		net = Network(sizes)

		evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, test_accuracy, test_confusion, epoch_num = \
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

	processed_resized = utils.createImage(net, test_data, test_osm, test_original)
	processed_accuracy = utils.evaluateResult(test_osm,processed_resized).tolist()

	"""Save the neural network info """
	data = {"sizes": sizes,
	"number of input images": len(im_numbers_train),
	"epochs": epochs,
	"mini batch size": mini_batch,
	"learning rate": learning_rate,
	"best evaluation result": test_accuracy,
	"evaluation result after process G": processed_accuracy,
	"achieved after epoch": epoch_num}

	f = open(save_dir + "/info", "w")
	json.dump(data, f)
	f.close()	

main()