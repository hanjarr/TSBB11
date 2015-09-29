from network import Network 
from osgeo import gdal
from matplotlib import pyplot as plt
from PIL import Image
import mnist_loader
import numpy as np
import scipy as sci
import random
import cv2
import image_slicer
import utils


def main():
	training_data, test_data = utils.load_data()
	net = Network([784,30,10])

	#print type(training_data)

	#print(np.shape(training_data))
	#print np.shape(training_data[1][0])

	#cv2.imshow("image window", vri)
	#cv2.waitKey()
	#net.SGD(training_data, 2, 10, 3.0, test_data)
main()