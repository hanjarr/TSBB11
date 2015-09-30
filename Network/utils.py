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

	tr_inputs = np.float32(vricon_arrays[:,0:7000])
	tr_inputs /= 255.0
	tr_keys = osm_arrays[:,0:7000]

	te_inputs = np.float32(vricon_arrays[:,7000:])
	te_inputs /= 255.0
	te_keys = osm_arrays[:,7000:]

	training_inputs = [np.reshape(tr_inputs[:,y],(400,1)) for y in range(0,7000)]
	training_keys = [class_label(tr_keys[:,x]) for x in range(0,7000)]
	training_data = zip(training_inputs,training_keys)

	test_inputs = [np.reshape(te_inputs[:,y],(400,1)) for y in range(0,3000)]
	test_keys= [test_label(te_keys[:,x]) for x in range(0,3000)]
	test_data = zip(test_inputs,np.int64(test_keys))

	#test_data = [[te_inputs[:,y],class_label(te_keys[:,y])] for y in range(0,3000)]

	return (training_data,test_data)


def class_label(label_array):
	mean = np.mean(label_array)
	class_array = np.zeros((3,1))
	if mean >250:
		class_array[0]=1.0
	elif mean < 200:
		class_array[1]=1.0
	else:
		class_array[2]=1.0
	return class_array

def test_label(label_array):
	mean = np.mean(label_array)
	if mean >200:
		class_label=0
	elif mean < 100:
		class_label=1
	else:
		class_label=2
	return class_label




