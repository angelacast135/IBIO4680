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

if not os.path.isdir("./for_CM"):
    os.system("mkdir for_CM")

if not os.path.isdir("./imgs_hists"):
    os.system("mkdir imgs_hists")


data = pickle.load( open( "./train_test_ims.pickle", "rb" ) )
im_train  = data['Train_Images']
im_test  = data['Test_Images']
print("Data loaded")

# Append the function provided in lib
sys.path.append('./lib/python')

#Create a filter bank with deafult params
from fbCreate import fbCreate

# return teh bank of filters with 8 orientations and 2 scales, it would be nice to add
# visualization of the bank of filters.
fb = fbCreate()


#Load sample images from disk
from skimage import color
from skimage import io

#Apply filterbank to sample image
from fbRun import fbRun

ACA_KNN=[]
ACA_RF=[]

kk = [25 , 40 , 60 , 80 ]
for k in enumerate(kk):
    maps_textons = pickle.load( open( './textons_k/maps_textons'+str(k[1])+'.pickle', "rb" ) )

    textons=maps_textons['Textons']
    print("Textons are loaded")

    im_test = np.reshape(im_test,(256,256,-1))
    im_train = np.reshape(im_train,(256,256,-1))
    # create texton maps over test
    imgs_test = []
    print("Textons map for test")

    from assignTextons import assignTextons
    for i in range(0,im_test.shape[2]):
        im1 =  assignTextons(fbRun(fb,im_test[:,:,i]),textons.transpose())
    
        imgs_test = imgs_test+[im1]
        print(i)

    # create texton maps over train
    imgs_train = []
    print("Textons map for train")

    for i in range(0,im_train.shape[2]):
        im2 =  assignTextons(fbRun(fb,im_train[:,:,i]),textons.transpose())
    
        imgs_train = imgs_train+[im2]
        print(i)

    with open('./imgs_hists/imgs_test'+str(k[1])+'.pickle', 'wb') as im3:
            pickle.dump(imgs_test, im3, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./imgs_hists/imgs_train'+str(k[1])+'.pickle', 'wb') as im4:
            pickle.dump(imgs_train, im4, protocol=pickle.HIGHEST_PROTOCOL)

    # hist function declaration
    def histc(X, bins):
        import numpy as np
        map_to_bins = np.digitize(X,bins)
        r = np.zeros(bins.shape)
        for i in map_to_bins:
            r[i-1] += 1
        return np.array(r)


    # create hist over train
    hist_train=[]
    print("Histograms for train")

    for i in range(0,im_train.shape[2]):
        hist1 = histc(imgs_train[i].flatten(), np.arange(k[1]))/imgs_train[i].size
    
        hist_train = hist_train+[hist1]
        print(i)


    # create hist over test
    hist_test=[]
    print("Histograms for test")

    for i in range(0,im_test.shape[2]):
        hist2 = histc(imgs_test[i].flatten(), np.arange(k[1]))/imgs_test[i].size
    
        hist_test = hist_test+[hist2]
        print(i)

    with open('./for_CM/hist_train'+str(k[1])+'.pickle', 'wb') as h:
            pickle.dump(hist_train, h, protocol=pickle.HIGHEST_PROTOCOL)


    with open('./for_CM/hist_test'+str(k[1])+'.pickle', 'wb') as h2:
            pickle.dump(hist_test, h2, protocol=pickle.HIGHEST_PROTOCOL)



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


    print("About to do KNN")

    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(weights='distance')
    neigh.fit(hist_train, labs_train) 
    predict_imgs_KNN=neigh.predict(hist_test)


    print("About to do RF")

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=15, random_state=0)
    clf.fit(hist_train, labs_train) 
    predict_imgs_RF=clf.predict(hist_test)
    
    import itertools
    
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
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
    cnf_matrix_KNN = confusion_matrix(labs_test, predict_imgs_KNN)
    np.set_printoptions(precision=2)

    cnf_matrix_KNN = cnf_matrix_KNN.astype('float') / cnf_matrix_KNN.sum(axis=1)[:, np.newaxis]
    n_accuracy_KNN=np.sum(cnf_matrix_KNN.diagonal())/25
    print("ACA for KNN: {}".format(n_accuracy_KNN))
    ACA_KNN=ACA_KNN+[n_accuracy_KNN]

    with open('./for_CM/conf_matrix_KNN'+str(k[1])+'.pickle', 'wb') as cf_m:
            pickle.dump(cnf_matrix_KNN, cf_m, protocol=pickle.HIGHEST_PROTOCOL)


    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix_KNN, classes=labels_total, normalize=True,
                          title='Normalized confusion matrix')

    #plt.show()
    plt.savefig('cm_KNN'+str(k[1])+'.eps', format='eps', dpi=1000)

    # Compute confusion matrix for RF
    cnf_matrix_RF = confusion_matrix(labs_test, predict_imgs_RF)
    np.set_printoptions(precision=2)

    cnf_matrix_RF = cnf_matrix_RF.astype('float') / cnf_matrix_RF.sum(axis=1)[:, np.newaxis]
    n_accuracy_RF=np.sum(cnf_matrix_RF.diagonal())/25
    print("ACA for RF: {}".format(n_accuracy_RF))
    ACA_RF=ACA_RF+[n_accuracy_RF]

    with open('./for_CM/conf_matrix_RF'+str(k[1])+'.pickle', 'wb') as cf_m2:
            pickle.dump(cnf_matrix_RF, cf_m2, protocol=pickle.HIGHEST_PROTOCOL)


    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix_RF, classes=labels_total, normalize=True,
                          title='Normalized confusion matrix')

    #plt.show()
    plt.savefig('cm_RF'+str(k[1])+'.eps', format='eps', dpi=1000)


f = open("ACA_KNN", "w")
f.write("\n".join(map(lambda x: str(x), ACA_KNN)) + "\n")
f.close()

f2 = open("ACA_RF", "w")
f2.write("\n".join(map(lambda x: str(x), ACA_RF)) + "\n")
f2.close()
