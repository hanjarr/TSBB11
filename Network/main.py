from network import Network 
from osgeo import gdal
from matplotlib import pyplot as plt
import mnist_loader
import numpy as np
import random
import cv2
import utils


def main():
	#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	#net = Network([784,30,10])

	block_dim = 10
	input_layer = block_dim**2
	hidden_layer = 10
	output_layer = 3

	training_data, test_data = utils.load_data(block_dim)
	net = Network([input_layer,hidden_layer,output_layer])

	#cv2.imshow("image window", vri)
	#cv2.waitKey()
	net.SGD(training_data, 1, 10, 10.0, test_data = test_data)
main()