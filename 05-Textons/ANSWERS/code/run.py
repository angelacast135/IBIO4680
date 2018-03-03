#!/usr/bin/env python3
import os
import time
import numpy as np
import random
import pickle
# matplotlib inline
import matplotlib.pyplot as plt
# OpenCV packages
import cv2
# for reading .mat files
import scipy.io as spio
import sys


if not os.path.isfile("./train_test_ims.pickle"):

    from read_n_download_dataset import read_n_download_dataset

    im_train , im_test = read_n_download_dataset()
    data_dict = {'Train_Images': im_train, 'Test_Images': im_test}
    with open('./train_test_ims.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    data = pickle.load( open( "./train_test_ims.pickle", "rb" ) )
    im_train  = data['Train_Images']
    im_test  = data['Test_Images']


# Append the function provided in lib
sys.path.append('../lib/python')

#Create a filter bank with deafult params
from fbCreate import fbCreate

# return teh bank of filters with 8 orientations and 2 scales, it would be nice to add
# visualization of the bank of filters.
fb = fbCreate()


#Load sample images from disk
from skimage import color
from skimage import io

#Set number of clusters
k = 25

#Apply filterbank to sample image
from fbRun import fbRun
a
im_train = np.reshape(im_train,(256,256,-1))
ims_to_textons = im_train[:,:,0]

print("This is the shape of the images to train {}".format(im_train.shape))

for i in range(1,im_train.shape[2]):
    im1 =  np.squeeze(ims_to_textons)
    im2 = np.squeeze(im_train[:,:,i])

    ims_to_textons = np.hstack( (im1 , im2)  )

print("This is the data size to obtain textons: {}".format(ims_to_textons.shape))



#images = np.hstack((imBase1,imBase2))
print("Running filter responses...")
filterResponses = fbRun(fb,ims_to_textons)


print("Saving filter responses...")
with open('./filter_responses.pickle', 'wb') as handle:
        pickle.dump(filterResponses, handle, protocol=pickle.HIGHEST_PROTOCOL)

## NUMBER OF TOTAL PIXEL FOR TRAIN 49,152,000


k = np.array([25 30 40 50 60 70 80 90 100])
for kk in enumerate(k):
    print("Computing textons from filter responses, consider k textons...")

#Computer textons from filter
    from computeTextons import computeTextons
    map, textons = computeTextons(filterResponses, kk[1])


    map_textons = {"Map", map, "Textons", textons}

    with open('./map_textons'+str(kk[1])+'.pickle', 'wb') as handle:
            pickle.dump(map_textons, handle, protocol=pickle.HIGHEST_PROTOCOL)



