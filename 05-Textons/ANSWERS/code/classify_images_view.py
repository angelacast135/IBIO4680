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

if not os.path.isdir("./cm_1"):
    os.system("mkdir cm_1")

data = pickle.load( open( "./train_test_ims.pickle", "rb" ) )
im_train  = data['Train_Images']
im_test  = data['Test_Images']
print("Data loaded")

#Load sample images from disk
from skimage import color
from skimage import io


ACA_KNN=[]
ACA_RF=[]

kk = [ 25, 40,60,80]
for k in enumerate(kk):
    im_train_4labs  = data['Train_Images']
    amount_classes_tr=im_train_4labs[0,0,0,:,0].shape
    amount_labs_tr=im_train_4labs[0,0,0,0,:].shape
    labels_total=('bark1','bark2','bark3', 'wood1', 'wood2', 'wood3', 'water', 'granite','marble', 'floor1', 'floor2', 'pebbles', 'wall', 'brick1', 'brick2', 'glass1', 'glass2', 'carpet1', 'carpet2', 'upholstery', 'wallpaper', 'fur', 'knit', 'corduroy', 'plaid')
    labs_train=[]
    for i in range(0, amount_classes_tr[0]):
        for j in range(0, amount_labs_tr[0]):
            labs_train = labs_train+[labels_total[i]]
    im_test_4labs  = data['Test_Images']
    amount_classes_ts=im_test_4labs[0,0,0,:,0].shape
    amount_labs_ts=im_test_4labs[0,0,0,0,:].shape
    labs_test=[]
    for i in range(0, amount_classes_ts[0]):
        for j in range(0, amount_labs_ts[0]):
            labs_test = labs_test+[labels_total[i]]
    import itertools
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=80)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(''), 
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix for KNN
    cnf_matrix_KNN = pickle.load( open( './for_CM/conf_matrix_KNN'+str(k[1])+'.pickle', "rb" ) )
    n_accuracy_KNN=np.sum(cnf_matrix_KNN.diagonal())/25
    print("ACA for KNN: {}".format(n_accuracy_KNN))
    ACA_KNN=ACA_KNN+[n_accuracy_KNN]
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix_KNN, classes=labels_total, normalize=True,
                         title='Normalized confusion matrix')
    # plt.show()
    plt.savefig('./cm_1/cm_KNN'+str(k[1])+'_1.eps', format='eps')

    # Compute confusion matrix for RF

    cnf_matrix_RF = pickle.load( open( './for_CM1/conf_matrix_RF'+str(k[1])+'.pickle', "rb" ) )
    n_accuracy_RF=np.sum(cnf_matrix_RF.diagonal())/25
    print("ACA for RF: {}".format(n_accuracy_RF))
    ACA_RF=ACA_RF+[n_accuracy_RF]

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix_RF, classes=labels_total, normalize=True,
                         title='Normalized confusion matrix')

    # plt.show()
    plt.savefig('./cm_1/cm_RF'+str(k[1])+'.eps', format='eps')


