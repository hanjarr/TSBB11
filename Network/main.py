from network import Network 
from osgeo import gdal
from matplotlib import pyplot as plt
from PIL import Image
import mnist_loader
import numpy as np
import scipy as sci
import random
import cv2
import utils


def main():
	#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	#net = Network([784,30,10])


	training_data, test_data = utils.load_data()
	net = Network([400,30,3])

	#print type(test_data)

	#print(np.shape(training_data))
	#print(np.shape(test_data))

	#print np.shape(test_data[0])
	#print np.shape(test_data[1])

	#print type(test_data)
	#print type(test_data[0])

	#print type(test_data[0][0])
	#print type(test_data[0][1])

	#print np.shape(test_data[0][1])
	#print np.shape(test_data[0][0])

	# print np.shape(training_data[0])
	# print np.shape(training_data[1])

	# print type(training_data)
	# print type(training_data[0])

	# print type(training_data[0][0])
	# print type(training_data[0][1])

	# print np.shape(training_data[0][1])
	# print np.shape(training_data[0][0])


	#cv2.imshow("image window", vri)
	#cv2.waitKey()
	net.SGD(training_data, 10, 10, 1.0, test_data = test_data)
main()