from network2 import Network 
from utils import Utils
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import cv2

def main():

	
	'''Load osm images for training and test (validation)'''
	training_osm = cv2.imread("../images/train_osm.png")
	test_osm =cv2.imread("../images/test_osm.png")

	'''Choose features for training and test'''

	training_features = "../python features/graylevels/f35_g128_b4_gau4_train.npy"
	test_features = "../python features/graylevels/f35_g128_b4_gau4_test.npy"

	info = map(int, re.findall(r'\d+', test_features))

	block_dim = info[2]
	input_layer = info[0]
	hidden_layer = 60
	output_layer = 3


	utils = Utils(block_dim, input_layer, output_layer)

	training_data, test_data = utils.load_data(training_osm[:,:,0], test_osm[:,:,0], training_features, test_features)
	net = Network([input_layer,hidden_layer,output_layer])

	evaluation_cost, evaluation_accuracy, \
	training_cost, training_accuracy = net.SGD(training_data, 100, 10, 0.008,lmbda=0.3, evaluation_data = test_data, \
	monitor_training_cost = True, monitor_evaluation_cost = True) #monitor_training_accuracy=True, monitor_evaluation_accuracy=True)
	
	if training_cost and evaluation_cost:
		plt.plot(training_cost)
		plt.plot(evaluation_cost)
		plt.savefig("cost.png")

	utils.createImage(net, test_data)
	
main()