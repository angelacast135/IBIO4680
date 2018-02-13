#!/usr/bin/env python
from pathlib import Path
import os
cwd = os.getcwd()

import tarfile

my_file = Path(cwd+'/BSR/BSDS500/data/images/test/2018.jpg')
if my_file.exists():
	print "The file is already downloaded"
else:
	import os
	print "The file is been downloaded"
	os.system('wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz')
	extracted_to_path = cwd+'/BSR_bsds500.tgz'
	tar = tarfile.open(extracted_to_path)
	tar.extractall()
	tar.close()

import time
start_time=time.time()

import scipy.io as spio

my_dir = cwd+'/BSR/BSDS500/data/images/test/'
my_dir_grtr = cwd+'/BSR/BSDS500/data/groundTruth/test/'
num_files = len([name for name in os.listdir(my_dir) if os.path.isfile(os.path.join(my_dir, name))])
import random

try:
	os.makedirs(cwd+'/resize/')
except OSError:
	if not os.path.isdir(cwd+'/resize/'):
		raise

from PIL import Image
import numpy
new_dir = cwd+'/resize/'
import matplotlib.pyplot as plt


list_files = [name for name in os.listdir(my_dir) if os.path.isfile(os.path.join(my_dir, name))]
# list_gt = [name for name in os.listdir(my_dir_grtr) if os.path.isfile(os.path.join(my_dir_grtr, name))]

big_struct = []

for k in range(0, 4):
	# IMAGES
	random_index = random.randrange(0,num_files)
	imageFile = my_dir+list_files[random_index]
	im1 = Image.open(imageFile)
	width = 256
	height = 256
	im2 = im1.resize((width, height), Image.BILINEAR) 
	im2.save(new_dir + list_files[random_index])
	# File
	myfile=list_files[random_index]
	name_file=myfile[0:len(list_files[random_index])-4]
	# GROUNDTRUTH
	gtFile = my_dir_grtr+name_file+'.mat'
	mat = spio.loadmat(gtFile, squeeze_me=True)
	M = mat['groundTruth']
	segg = M[0]['Segmentation'][()][:,:]
	bounn = M[0]['Boundaries'][()][:,:]
	big_struct = big_struct+[im1,segg,bounn]

import pickle 
with open('objs.pkl', 'w') as f:
	pickle.dump([big_struct], f)

plt.figure(1)
for k in range(1, 13):
	img=big_struct[k-1]
	plt.subplot(4, 3, k)
	plt.imshow(img)

final_time = time.time() - start_time
print "Total processing time is %d seconds" % (final_time)
plt.show()