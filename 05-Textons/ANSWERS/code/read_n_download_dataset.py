import glob, os
import numpy as np
import numpy.matlib


# matplotlib inline
import matplotlib.pyplot as plt
# OpenCV packages
import cv2


def read_n_download_dataset():
    if not os.path.isdir("./data"):
        os.system('mkdir ./data')
        print('Connecting and downloading Texture Database from Ponce Group...')
        print('1st textures...')
        os.system('wget http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T01-T05.zip')
        print('2nd textures...')
        os.system('wget http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T06-T10.zip')
        print('3rd textures...')
        os.system('wget http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T11-T15.zip')
        print('4th textures...')
        os.system('wget http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T16-T20.zip')
        print('5th textures...')
        os.system('wget http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T21-T25.zip')

        print('Unziping files...')
        os.system('unzip ./T01-T05.zip -d ./data')
        os.system('rm T01-T05.zip ')
        os.system('unzip ./T06-T10.zip -d ./data')
        os.system('rm T06-T10.zip ')
        os.system('unzip ./T11-T15.zip -d ./data')
        os.system('rm T11-T15.zip ')
        os.system('unzip ./T16-T20.zip -d ./data')
        os.system('rm T16-T20.zip ')
        os.system('unzip ./T21-T25.zip -d ./data')
        os.system('rm T21-T25.zip ')

    else:
        print('Texture dataset is already downloaded, proceding to create the filter bank')
    

    #ims = np.uint8(np.empty((480,640,1,40*25)))
    ims = np.uint8(np.empty((256,256,1,40*25)))

    i=0
    for dirs in os.listdir('./data'):
        
        if not dirs.startswith('.'):
            for file in glob.glob(os.path.join("./data",dirs,"*.jpg")):

                #print("this is the image to read {}".format(file))
                #read image in grayscale
                image = cv2.imread(file,0)
                #crop original image size (480,640) from center to (256,256)
                #a = int(image.shape[0]/2-256/2):int(image.shape[0]/2+256/2)
                ims[:,:,0,i]=image[ int(image.shape[0]/2-256/2):int(image.shape[0]/2+256/2) , int(image.shape[0]/2-256/2):int(image.shape[0]/2+256/2) ]
                i+=1


    ims = np.reshape(ims,(256,256,1,25,40))


    #return ims




    def rand_idx_test(k_classes,ims_test,im_perclass):
        import random

        idx = np.empty((k_classes,ims_test))
        for j in range(0,k_classes):

            idx[j,:] = np.array(random.sample(range(im_perclass), ims_test))
    
        return np.int8(idx)


    def random_sample_for_test(ims):


        ims_train = np.uint8(np.empty((256,256,1,25,30)))
        ims_test = np.uint8(np.empty((256,256,1,25,10)))

    # split data into 10 images per test randomly for 25 classes, 10 images per calass in test and a total of 40 images per class
        idx = rand_idx_test(25,10,40)

        k = 1
        for i in range(0,25):
            ims_test[:,:,:,i,:] = ims[:,:,:,i,idx[i]]

            ims_train = np.delete(ims,idx[i],4)
        return ims_train , ims_test 


    #ims = read_n_download_dataset()

    im_train , im_test = random_sample_for_test(ims)

    return im_train , im_test



