from network2 import Network 
from utils4 import Utils
from osgeo import gdal
from matplotlib import pyplot as plt
import mnist_loader
import numpy as np
import random
import cv2

def main():

	block_dim = 4
	input_layer = 28
	hidden_layer = 28
	output_layer = 3


	utils = Utils(block_dim, input_layer, output_layer)

	training_data, test_data = utils.load_data()
	net = Network([input_layer,hidden_layer,output_layer])

	net.SGD(training_data, 10, 10, 0.05,lmbda=0.3, evaluation_data = test_data, monitor_training_accuracy=True, monitor_evaluation_accuracy=True)
	
	utils.createImage(net, test_data)
	
main()