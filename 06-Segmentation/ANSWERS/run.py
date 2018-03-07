
 ################## This is run.py for Segmentation assignenmt of IBIO4680 ################## 

import os
import time
import numpy as np
import random
import pickle
import glob

# matplotlib inline
import matplotlib.pyplot as plt
# OpenCV packages
# normal installation routine


import cv2
# for reading .mat files
import scipy.io as spio
from skimage import io, color



def read_groundtruth(mat_file_name , version_human ):

    a = spio.loadmat('../BSDS_tiny/'+mat_file_name)['groundTruth']


    segm = a[0,[1]][0]['Segmentation'][0,0]
    bound = a[0,[1]][0]['Boundaries'][0,0]


    # for i in range(1 , a.shape[1]):
    #     segm = segm + a[0,[i]][0]['Segmentation'][0,0]
    #     bound = bound + a[0,[i]][0]['Boundaries'][0,0]

    #segm = segm + a[0,[version_human]][0]['Segmentation'][0,0]
    #bound = bound + a[0,[version_human]][0]['Boundaries'][0,0]

    segm = np.uint8(segm)
    segm = segm[ int(segm.shape[0]/2-256/2):int(segm.shape[0]/2+256/2) , int(segm.shape[1]/2-256/2):int(segm.shape[1]/2+256/2) ]

    bound = np.uint8(bound)

    bound = bound[ int(bound.shape[0]/2-256/2):int(bound.shape[0]/2+256/2) , int(bound.shape[1]/2-256/2):int(bound.shape[1]/2+256/2) ]
 
    return segm, bound

def read_dataset():
    ims = np.uint8(np.empty((256,256,3,8*3)))
    for i, file in enumerate(glob.glob(os.path.join("../BSDS_tiny/*.jpg"))):
        
        image =cv2.imread(file,1) 
        image = image[ int(image.shape[0]/2-256/2):int(image.shape[0]/2+256/2) , int(image.shape[1]/2-256/2):int(image.shape[1]/2+256/2) ]
        #if image.shape[0] == 481:
        #    ims[:,:,:,i] =  image
        #else :
        #    ims[:,:,:,i] = np.transpose(image , (1, 0, 2))
    segm = np.uint8(np.empty((256,256,8*3)))
    bound = np.uint8(np.empty((256,256,8*3)))

    for i , file in enumerate(glob.glob(os.path.join("../BSDS_tiny/*.mat"))):
        segm[ : , : , i ], bound[ : , : , i ] = read_groundtruth(file,3)

    return ims, segm, bound

def k_means( im , k  ):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn import preprocessing

    im_size = im.shape

    #reshape image conserving colour channels and given the case x,y features
    im_vector=np.reshape(im, (-1,im_size[-1]))

    #now normalize image for posing the problem, this scale transform x\in R^n  with \mu_x=0, std_x=1 
    im_vector = preprocessing.scale(im_vector)
            
    #print(k)
    #print(type(k))
    print(np.int8(k))
    #print(type(np.int8(k)))

    k_im = KMeans(n_clusters=np.uint8(k), init='random').fit( im_vector ) 
    im_segmented = k_im.predict(im_vector)
    im_segmented = np.reshape(im_segmented , (im_size[0],im_size[1]) )

    return im_segmented

def GMM_im( im , k  ):
    import numpy as np
    from sklearn.mixture import GMM
    from sklearn import preprocessing

    im_size = im.shape
    #reshape image conserving colour channels and given the case x,y features
    im_vector=np.reshape(im, (-1,im_size[-1]))
    #now normalize image for posing the problem, this scale transform x\in R^n  with \mu_x=0, std_x=1 
    im_vector = preprocessing.scale(im_vector)
            

    GMM_ims = GMM(n_components=np.int8(k), covariance_type='full',init='random').fit( im_vector ) 
    im_segmented = GMM_ims.predict(im_vector)
    im_segmented = np.reshape(im_segmented , (im_size[0],im_size[1]) )

    return im_segmented

def hierarchical_segmentation(im, n_clusters):
    import numpy as np
    import scipy as sp
    from sklearn import preprocessing
    from sklearn.feature_extraction.image import grid_to_graph
    from sklearn.cluster import AgglomerativeClustering

    im_size = im.shape
    im_vector = np.zeros((im_size[0]*im_size[1],im_size[2]))

    #reshape image conserving colour channels
    im_vector=np.reshape(im, (-1,im_size[-1]))

    #now normalize image for posing th problem, this scale transform x\in R^n \mu_x=0, std_x=1 
    im_vector = preprocessing.scale(im_vector)
    
    #X = np.reshape(im_vector, (-1, 1))

    connectivity = grid_to_graph(*im_vector.shape)
    ward = AgglomerativeClustering(n_clusters=n_clusters)
    ward.fit(im_vectors)
    im_segmented = np.reshape(ward.labels_ , (im_size[0],im_size[1]) )

    return im_segmented

def add_position_feature(image):
    im_size = image.shape
    x_idx = np.array(range(im_size[0]))
    x_idx = np.matlib.repmat(x_idx, im_size[1],1)
    x_idx = np.transpose(x_idx)

    y_idx = np.array(range(im_size[1]))
    y_idx = np.matlib.repmat(y_idx, im_size[0] , 1)

    im_xy = np.dstack((image,x_idx,y_idx))

    return im_xy


#numberOfClusters positvie integer (larger than 2)
def segmentByClustering( rgbImage, featureSpace, clusteringMethod, numberOfClusters):
    

    #featureSpace : 'rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy' or 'hsv+xy'
    if featureSpace == 'rgb':
        im_feat = rgbImage
    
    elif featureSpace == 'lab': 
        im_feat = color.rgb2lab(rgbImage)

    elif featureSpace == 'hsv':
        im_feat = color.rgb2hsv(rgbImage)

    elif featureSpace == 'rgb+xy':
        im = rgbImage
        im_feat = add_position_feature(im)

    elif featureSpace == 'lab+xy':
        im = color.rgb2lab(rgbImage)
        im_feat = add_position_feature(im)


    elif featureSpace == 'hsv+xy':
        im = color.rgb2hsv(rgbImage)
        im_feat = add_position_feature(im)

    else:
        # default assume feature space is LAB 
        featureSpace = 'lab'
        im_feat = color.rgb2lab(rgbImage)


    #clusteringMethod = 'kmeans', 'gmm', 'hierarchical' or 'watershed'.

    if clusteringMethod == 'kmeans':
        #return vector of segmentaion 
        im_segmented = k_means(im_feat,numberOfClusters)
    
    elif clusteringMethod == 'gmm': 
        im_segmented = GMM_im(im_feat,numberOfClusters)


    elif clusteringMethod == 'hierarchical':
        im_segmented = hierarchical_segmentation(im_feat,numberOfClusters)

    #elif clusteringMethod == 'watershed':
    
    else:
        #by default assume kmeans is used
        clusteringMethod == 'kmeans'
        im_seg = k_means(im_feat,numberOfClusters)


    return im_segmented , clusteringMethod , featureSpace

if  os.path.isdir("../BSDS_tiny"):
    print('Reading tiny version of BSDS ...')
    ims, segm, bound = read_dataset()
else:
    print('BSDS500 dataset is downlading...')
    os.system('wget http://157.253.63.7/BSDS_tiny.zip')
    os.system('mv ./BSDS_tiny.zip ../BSDS_tiny.zip')
    os.system('unzip ../BSDS_tiny')
    os.system('mv  ./BSDS_tiny ../BSDS_tiny')
    os.system('rm ../BSDS_tiny.zip')


featureSpaces =   [ 'rgb','lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy']
clusteringMethods = ['kmeans','gmm','hierarchical']

for ff , featureSpace in enumerate(featureSpaces):
    for cc, clusteringMethod in enumerate(clusteringMethods):

        for i in range( 0,5 ):
            im = ims[:,:,:,i]
            plt.imshow(np.squeeze(im))

            seg = segm[:,:,i]

            numberOfClusters = np.unique(np.squeeze(seg)).shape[0]
            print("The number of cluster in current image is {}".format(numberOfClusters))

            

            eval_seg = {"Im_segmented": im, "Segmentation_GT": seg}

            im_seg , clusteringMethod, featureSpace = segmentByClustering( np.double(im), featureSpace, clusteringMethod , numberOfClusters)

            fig1=plt.subplot(1,2,1)
            fig1=plt.imshow(np.squeeze(seg), cmap=plt.get_cmap('summer'))
            fig1.axes.get_xaxis().set_visible(False)
            fig1.axes.get_yaxis().set_visible(False)

            fig1=plt.subplot(1,2,2)
            fig1=plt.imshow(np.uint8((im_seg)), cmap=plt.get_cmap('summer'))
            fig1.axes.get_xaxis().set_visible(False)
            fig1.axes.get_yaxis().set_visible(False)
            fig1=plt.show()
            #plt.imshow(np.squeeze(im_seg), cmap=plt.get_cmap('summer'))
            #plt.show()
            print('File to save: '+'./'+featureSpace+'_'+clusteringMethod+'/eval_segIm'+str(i)+'_K'+str(numberOfClusters)+'.pickle')
            with open('./'+featureSpace+'_'+clusteringMethod+'/eval_segIm'+str(i)+'_K'+str(numberOfClusters)+'.pickle', 'wb') as handle:
                pickle.dump(eval_seg, handle, protocol=pickle.HIGHEST_PROTOCOL)

