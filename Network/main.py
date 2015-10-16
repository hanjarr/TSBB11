from network import Network 
from osgeo import gdal
from matplotlib import pyplot as plt
import mnist_loader
import numpy as np
import random
import cv2
import utils3


def main():

	block_dim = 4
	input_layer = 4
	hidden_layer = 2
	output_layer = 3

	training_data, test_data = utils3.load_data(block_dim)
	net = Network([input_layer,hidden_layer,output_layer])

	net.SGD(training_data, 10, 10, 100.0, test_data = test_data)
main()