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

sys.path.append('../lib/python')

data = pickle.load( open( "./train_test_ims.pickle", "rb" ) )
im_train  = data['Train_Images']
im_test  = data['Test_Images']
print(im_train.shape)
fig=plt.figure()
for n in range(0,25):
    img = np.uint8(np.squeeze( im_train[:,:,:,n,5] ))

    #fig=plt.subplot(5,5,  n+1)
    #fig=plt.imshow(img,cmap = 'gray')
    #fig.axes.get_xaxis().set_visible(False)
    #fig.axes.get_yaxis().set_visible(False)

print('Showing images')
#fig=plt.show()

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


col_1 = np.pad(np.uint8(np.squeeze( im_train[:,:,:,0,2] )),4,pad_with ,padder=1)
col_2 = np.pad(np.uint8(np.squeeze( im_train[:,:,:,0+5,2] )),4,pad_with ,padder=1)
col_3 = np.pad(np.uint8(np.squeeze( im_train[:,:,:,0+10,2] )),4,pad_with ,padder=1)
col_4 = np.pad(np.uint8(np.squeeze( im_train[:,:,:,0+15,2] )),4,pad_with ,padder=1)
col_5 = np.pad(np.uint8(np.squeeze( im_train[:,:,:,0+20,2] )),4,pad_with ,padder=1)

for i in range(1,5):
    col_1 = np.vstack([col_1, np.pad(np.uint8(np.squeeze( im_train[:,:,:,i,2] )) ,4,pad_with ,padder=1)])
    col_2 = np.vstack([col_2, np.pad(np.uint8(np.squeeze( im_train[:,:,:,i+5,2] )),4,pad_with ,padder=1)])
    col_3 = np.vstack([col_3, np.pad(np.uint8(np.squeeze( im_train[:,:,:,i+10,2] )),4,pad_with ,padder=1)])
    col_4 = np.vstack([col_4, np.pad(np.uint8(np.squeeze( im_train[:,:,:,i+15,2] )),4,pad_with ,padder=1)])
    col_5 = np.vstack([col_5, np.pad(np.uint8(np.squeeze( im_train[:,:,:,i+20,2] )),4,pad_with ,padder=1)])

#img = np.hstack([col_1,col_2,col_3,col_4,col_5])
#fig=plt.imshow(np.transpose(img),cmap = 'gray')
#fig.axes.get_xaxis().set_visible(False)
#fig.axes.get_yaxis().set_visible(False)
#fig=plt.show()


from fbCreate import fbCreate

# return teh bank of filters with 8 orientations and 2 scales, it would be nice to add
# visualization of the bank of filters.
fb = fbCreate()
fb=np.array(fb)
#(16, 2)
#print(fb[0,1])



fb1 = fb[0,0] 
fb1 = 255*(fb1-fb1.min())/(fb1.max()-fb1.min())
fb1 = np.pad(fb1,5,pad_with ,padder=1)

fb2 = fb[0,1]
fb2 = 255*(fb2-fb2.min())/(fb2.max()-fb2.min())
fb2 = np.pad(fb2,2,pad_with ,padder=1)


for i in range(1,16):
    fb11 = fb[i,0]
    fb11 = 255*(fb11-fb11.min())/(fb11.max()-fb11.min())

    fb22 = fb[i,1]
    fb22 = 255*(fb22-fb22.min())/(fb22.max()-fb22.min())

    fb1 = np.hstack([fb1, np.pad(np.uint8(np.squeeze( fb11 )) ,5,pad_with ,padder=1)])
    fb2 = np.hstack([fb2, np.pad(np.uint8(np.squeeze( fb22 )),2,pad_with ,padder=1)])

fb = np.vstack([fb1,fb2])
fb = 255*(fb-fb.min())/(fb.max()-fb.min())


fig1 = plt.imshow(fb,cmap = 'gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
fig1=plt.figure()
fig1=plt.show()



