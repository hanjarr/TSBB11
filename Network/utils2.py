import numpy as np 
import cv2
from skimage.feature import greycomatrix, greycoprops

def split_image(image_array, block_dim, vricon_data = False):

	image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

	image_dim = np.shape(image_array);
	num_blocks = image_dim[0]**2/(block_dim**2)
	image_blocks = np.zeros((block_dim,block_dim,num_blocks))

	if vricon_data:
		feature_blocks = np.zeros((4,4,num_blocks))

	iterations = image_dim[0]/block_dim

	i = 0
	for k in range(0,iterations):
	    for j in range(0,iterations):
	    	image_blocks[:,:,i] = image_array[block_dim*k:block_dim+block_dim*k,block_dim*j:block_dim+block_dim*j]
	    	if vricon_data:
	    		feature_blocks[:,:,i] = im_stats(image_blocks[:,:,i])
	    	i += 1
	if vricon_data:
		return feature_blocks
	else:
		return image_blocks


def load_data(block_dim):

	np.set_printoptions(threshold=np.nan)

	osm = cv2.imread("osm.png")
	vricon = cv2.imread("vri.png")

	osm_blocks = split_image(osm, block_dim)
	vricon_blocks = split_image(vricon, block_dim, vricon_data =True)

	image_dim = np.shape(vricon);
	num_blocks = image_dim[0]**2/(block_dim**2)
	training_blocks = int(num_blocks*0.8)
	test_blocks = num_blocks - training_blocks


	# vricon_arrays is a [16,25000] ndarray (if block_size=4) and vricon_array[:,x] contains pixel values for block x
	# should be replaced with a feature matrix where every row contains features for the corresponding image block

	# CHANGE IN RESHAPE OF VRICON ARRAY, TR_INPUTS AND TE_INPUTS TO NUMBER OF FEATURES

	vricon_arrays = np.reshape(vricon_blocks, (block_dim**2,num_blocks))
	osm_arrays = np.uint8(np.reshape(osm_blocks, (block_dim**2,num_blocks)))

	tr_inputs = np.float32(vricon_arrays[:,0:training_blocks])
	tr_keys = osm_arrays[:,0:training_blocks]

	te_inputs = np.float32(vricon_arrays[:,training_blocks:])
	te_keys = osm_arrays[:,training_blocks:]

	training_inputs = [np.reshape(tr_inputs[:,y],(block_dim**2,1)) for y in xrange(0,training_blocks)]
	training_keys = [training_label(tr_keys[:,x]) for x in xrange(0,training_blocks)]
	training_data = zip(training_inputs,training_keys)

	test_inputs = [np.reshape(te_inputs[:,y],(block_dim**2,1)) for y in xrange(0,test_blocks)]
	test_keys= [test_label(te_keys[:,x]) for x in xrange(0,test_blocks)]
	test_data = zip(test_inputs,np.int64(test_keys))

	return (training_data,test_data)


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

def im_stats(image_block):
	pixel_pairs = greycomatrix(image_block, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed =True)
	measures = ['contrast','correlation','energy','homogeneity']
	feature_arrays = np.zeros((4,4))

	for index,measure in enumerate(measures):
		feature_arrays[:,index] = greycoprops(pixel_pairs,measure)
	return feature_arrays
