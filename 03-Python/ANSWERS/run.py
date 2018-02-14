#!/usr/bin/env python3

import os
import time
import numpy as np
import random
import pickle

# matplotlib inline
import matplotlib.pyplot as plt
# OpenCV packages
# normal installation routine


import cv2
# for reading .mat files
import scipy.io as spio


# Downlaod BSDS500 image segmentation dataset from Computer Vision - Berkley group, decompress it and remove .tgz file.
tic1 = time.clock()
if not os.path.isdir("./BSR"):
    print('Connecting and downloading BSR dataset...')
    os.system('wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz')
    os.system('tar -zxvf BSR_bsds500.tgz')
    os.system('rm BSR_bsds500.tgz')
else:
    tic2 = time.clock()
    print('BSDS500 dataset is already downloaded')

# cre el path con el directorio para guardar las nuevas imagenes 
if not os.path.isdir("./resized_images"):
    os.system('mkdir ./resized_images')
else :
    os.system('rm -r ./resized_images')

dataPath = './BSR/BSDS500/data'

def read_images(path_to_images):
  im_list = os.listdir(path_to_images)
  return im_list

# list with .jpg name files. For reaning any other subset images change .../train to .../test or .../validation
train_images = read_images(dataPath+'/images/train')
# list with .mat name files.
train_gt = read_images(dataPath+'/groundTruth/train')

# N number of images that will be loaded, assume that can plot from 7 to 10 images for descent visualization
N = random.randint(7, 10) 
print('Number of images to plot: {}'.format(N))
print('Number of images in train:  {}'.format(len(train_images)-1))


images_idx = np.random.randint(len(train_images)-1,size=N)
images_idx.sort()

#preallocation for computational time
ims = np.uint8(np.empty((256, 256,3,N)))
segmentation_gt = np.empty((256, 256,N))
boundary_gt = np.empty((256, 256,N))



def read_groundtruth(mat_file_name,resized_size):
    a = spio.loadmat(dataPath+'/groundTruth/train/'+mat_file_name)['groundTruth']


    segm = a[0,[0]][0]['Segmentation'][0,0]
    bound = a[0,[0]][0]['Boundaries'][0,0]


    for i in range(1 , a.shape[1]):
        segm = segm + a[0,[i]][0]['Segmentation'][0,0]
        bound = bound + a[0,[i]][0]['Boundaries'][0,0]
    segm = np.uint8(segm)
    bound = np.uint8(bound)

    segm = cv2.resize(segm , resized_size,interpolation = cv2.INTER_CUBIC)
    #interpolation for preserving better connected boundaries
    bound = cv2.resize(bound , resized_size,interpolation = cv2.INTER_CUBIC)
    return segm, bound

for ix, i in enumerate(images_idx):
    
    # append the images in a tensor 4 dimension correspond to each image. 
    #ims(ix) = np.append( ims , [img] , axis=0)
    ims[:,:,:,ix] =cv2.resize( cv2.imread(dataPath+'/images/train/'+train_images[i],1), (256,256),interpolation = cv2.INTER_CUBIC)

    cv2.imwrite('./resized_images/'+train_images[i],np.squeeze(ims[:,:,:,ix]))



    # save segmentation and boundaries in segmentation_gt and boundary_gt respectively with all the annotations (from every annotator)
    segmentation_gt[:,:,ix] , boundary_gt[:,:,ix] = read_groundtruth(train_gt[i],(256,256))

#print('Size of Original Images {}, Sementation {} and Boundaries {}'.format(ims.shape,segmentation_gt.shape,boundary_gt.shape))
    

def show_images(images , boundary , segmentation, cols ):
  
    fig=plt.figure()
    for n in range(0,cols):
        img = np.uint8(np.squeeze( images[:,:,:,n] ))

        fig=plt.subplot(3,cols,  n+1)
        fig=plt.imshow(img[:,:,[2,1,0]])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        img = np.uint8(np.squeeze( segmentation[:,:,:,n] ))
        fig =plt.subplot(3,cols,  n+cols+1)
        fig =plt.imshow(img, cmap = 'gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        img = np.uint8(np.squeeze( boundary[:,:,:,n] ))
        fig = plt.subplot(3,cols, n+2*cols+1)
        fig = plt.imshow(img , cmap = 'gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    print('Showing images')
    fig=plt.show()

segmentation_gt = np.expand_dims(segmentation_gt , axis=2)
boundary_gt = np.expand_dims(boundary_gt , axis=2)

data_dict = {'Images': ims, 'Segmentation': segmentation_gt, 'Boundaries': boundary_gt}
with open('./data_dict.pickle', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

toc = time.clock()
show_images(ims , boundary_gt , segmentation_gt , N)



print('Elapsep time downloading dataset: {}; Elapsep time without downloading dataset: {}'.format(toc-tic1,toc-tic2))






