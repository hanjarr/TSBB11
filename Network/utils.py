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


def load_data(block_dim):

	osm = cv2.imread("osm.png")
	vricon = cv2.imread("vri.png")
	osm_blocks = split_image(osm, block_dim)
	vricon_blocks = split_image(vricon, block_dim)

	image_dim = np.shape(vricon);
	num_blocks = image_dim[0]**2/(block_dim**2)
	training_blocks = int(num_blocks*0.8)
	test_blocks = num_blocks - training_blocks

	vricon_arrays = np.reshape(vricon_blocks, (block_dim**2,num_blocks))
	osm_arrays = np.reshape(osm_blocks, (block_dim**2,num_blocks))

	tr_inputs = np.float32(vricon_arrays[:,0:training_blocks])
	tr_inputs /= 255.0
	tr_keys = osm_arrays[:,0:training_blocks]

	te_inputs = np.float32(vricon_arrays[:,training_blocks:])
	te_inputs /= 255.0
	te_keys = osm_arrays[:,training_blocks:]

	training_inputs = [np.reshape(tr_inputs[:,y],(block_dim**2,1)) for y in range(0,training_blocks)]
	training_keys = [training_label(tr_keys[:,x]) for x in range(0,training_blocks)]
	training_data = zip(training_inputs,training_keys)

	test_inputs = [np.reshape(te_inputs[:,y],(block_dim**2,1)) for y in range(0,test_blocks)]
	test_keys= [test_label(te_keys[:,x]) for x in range(0,test_blocks)]
	test_data = zip(test_inputs,np.int64(test_keys))

	return (training_data,test_data)


# def class_label(label_array):
# 	mean = np.mean(label_array)
# 	class_array = np.zeros((3,1))
# 	if mean >250:
# 		class_array[0]=1.0
# 	elif mean < 100:
# 		class_array[1]=1.0
# 	else:
# 		class_array[2]=1.0
# 	return class_array

def training_label(label_array):
	class_array = np.zeros((3,1))
	counts = np.bincount(label_array)
	block_class = np.argmax(counts)
	if block_class == 255:
		class_array[0]=1.0
	elif block_class == 232:
		class_array[1]=1.0
	else:
		class_array[2]=1.0
	return class_array


def test_label(label_array):
	counts = np.bincount(label_array)
	block_class = np.argmax(counts)
	if block_class == 255:
		class_label=0
	elif block_class == 232:
		class_label=1
	else:
		class_label=2
	return class_label

# def test_label(label_array):
# 	mean = np.mean(label_array)
# 	if mean >250:
# 		class_label=0
# 	elif mean < 100:
# 		class_label=1
# 	else:
# 		class_label=2
# 	return class_label




