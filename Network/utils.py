import numpy as np
import cv2

def split_image(image_array, block_dim):

	image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
	
	image_dim = np.shape(image_array);
	num_blocks = image_dim[0]**2/(block_dim**2)
	image_blocks = np.zeros((block_dim, block_dim, num_blocks))

	iterations = image_dim[0]/block_dim

	i = 0
	for k in range(0,iterations):
	    for j in range(0,iterations):
	    	image_blocks[:,:,i] = image_array[block_dim*k:block_dim+block_dim*k,block_dim*j:block_dim+block_dim*j]
	    	i += 1
	image_blocks = np.uint8(image_blocks)
	return image_blocks


def load_data():

	np.set_printoptions(threshold=np.nan)

	osm = cv2.imread("osm.png")
	vricon = cv2.imread("vri.png")
	osm_blocks = split_image(osm, 20)
	vricon_blocks = split_image(vricon, 20)

	vricon_arrays = np.reshape(vricon_blocks, (400,10000))
	osm_arrays = np.reshape(osm_blocks, (400,10000))


	data = [[vricon_arrays[:,y],class_label(osm_arrays[:,y])] for y in range(0,10000)]

	print np.shape(data)


def class_label(label_array):
	mean = np.mean(label_array)
	if mean >250:
		return np.array(1,0,0)
	elif mean < 200:
		return np.array(0,0,1)
	else:
		return np.array(0,1,0)




