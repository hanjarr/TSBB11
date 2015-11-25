import numpy as np
import sys

#Load saved feature
def appendFeature(input_array1, input_array2, output_array):
    array1 = np.load(input_array1)
    array2 = np.load(input_array2)
    print array1.shape, array2.shape
    array_length1 = array1.shape[1]
    array_length2 = array2.shape[1]
    if array_length1 != array_length2:
        print "Error: lenght do not mach"
        sys.exit()
    combined_array = np.append(array1,array2,axis=0)
    print combined_array.shape
    np.save(output_array, combined_array)

    #Check is reshaped image is correct
    # ret_image = []
    # ret_array = feature_array[-1,:]
    # im_ret = np.reshape(ret_array,[250,250])
    #print im_corr.shape

input_array1 = "graylevels/f35_g128_b4_gau4_train.npy"
input_array2 =  "testKarin3.npy"
output_array = "outputArrayTest.npy"

appendFeature(input_array1,input_array2,output_array)
