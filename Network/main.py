import network2
from network2 import Network 
from utils import Utils
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import os
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

	hidden_layer = 50
	hidden_layer_2 = 20
	output_layer = 3

	epochs = 50
	mini_batch = 10
	learning_rate = 0.01
	save_net = False
	load_net = False

	'''Load existing network from folder svaved_network'''
	if load_net:
		network2.load("../saved_networks/..")

	utils = Utils(block_dim, input_layer, output_layer)

	training_data, test_data = utils.loadData(training_osm[:,:,0], test_osm[:,:,0], training_features, test_features)
	net = Network([input_layer,hidden_layer, output_layer],cost=network2.CrossEntropyCost)

	evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
	net.SGD(training_data, epochs, mini_batch, learning_rate,lmbda=0.5, evaluation_data = test_data, \
	monitor_training_cost = True, monitor_evaluation_cost = True, monitor_training_accuracy=True, monitor_evaluation_accuracy=True)

	''' Save network '''
	if save_net:
		net.save("../saved_networks/e"+str(epochs)+"_mb"+str(mini_batch)+"_lr"+str(learning_rate)+os.path.basename(training_features))

	''' Create accuracy plot'''
	if evaluation_accuracy:
		others = evaluation_accuracy[0::3]
		water = evaluation_accuracy[1::3]
		roads = evaluation_accuracy[2::3]

		num_last = 5

		acc_others = sum(others[-num_last:])/num_last
		acc_water = sum(water[-num_last:])/num_last
		acc_roads = sum(roads[-num_last:])/num_last

		print('ACCURACY OTHERS: ' + str(acc_others) + '\n')
		print('ACCURACY WATER: ' + str(acc_water) + '\n')
		print('ACCURACY ROADS: ' + str(acc_roads) + '\n')

		#line1, = plt.plot(others, 'r')
		#line2, = plt.plot(water, 'g')
		#line3, = plt.plot(roads, 'b')
		#plt.ylabel('Accuracy')
		#plt.xlabel('Epochs')
		#plt.legend([line1,line2,line3],["Others", "Water", "Roads"])
		#plt.savefig("../saved_cost/accuracy_e"+str(epochs)+"_mb"+str(mini_batch)+"_lr"+str(learning_rate)+".png")
		#plt.clf()

	''' Create cost plot'''
	if training_cost and evaluation_cost:

		final_cost = sum(evaluation_cost[-5:])/5

		print('CROSS ENTROPY COST: ' + str(final_cost))
		#line1, = plt.plot(training_cost)
		#line2, = plt.plot(evaluation_cost)
		#plt.ylabel('Cost')
		#plt.xlabel('Epochs')
		#plt.legend([line1,line2],["Training", "Evaluation"])
		#plt.savefig("../saved_cost/cost_e"+str(epochs)+"_mb"+str(mini_batch)+"_lr"+str(learning_rate)+".png")

	utils.createImage(net, test_data)
	
main()