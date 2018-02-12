
from pathlib import Path
import os
cwd = os.getcwd()


# my_file = Path(cwd+'/BSR/BSDS500/data/images/test/2018.jpg')
# if my_file.exists():
# 	print "Here I Am"
# 	# file exists
# else:
# 	import os
# 	os.system('wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz')
# 	extracted_to_path = cwd+'/BSR_bsds500.tgz'
# 	tar = tarfile.open(extracted_to_path)
# 	tar.extractall()
# 	tar.close()

import scipy.io as spio

my_dir = cwd+'/BSR/BSDS500/data/images/test/'
my_dir_grtr = cwd+'/BSR/BSDS500/data/groundTruth/test/'
num_files = len([name for name in os.listdir(my_dir) if os.path.isfile(os.path.join(my_dir, name))])
# import random
# random_index = random.randrange(0,num_files)

# try:
# 	os.makedirs(cwd+'/resize/')
# except OSError:
# 	if not os.path.isdir(cwd+'/resize/'):
# 		raise

from PIL import Image
import numpy
new_dir = cwd+'/resize/'
import matplotlib.pyplot as plt


list_files = [name for name in os.listdir(my_dir) if os.path.isfile(os.path.join(my_dir, name))]
list_gt = [name for name in os.listdir(my_dir_grtr) if os.path.isfile(os.path.join(my_dir_grtr, name))]

big_struct = []

for k in range(0, random_index):
	# IMAGES
	imageFile = my_dir+list_files[k]
	im1 = Image.open(imageFile)
	width = 256
	height = 256
	im2 = im1.resize((width, height), Image.BILINEAR) 
	im2.save(new_dir + list_files[k])
	# GROUNDTRUTH
	gtFile = my_dir_grtr+list_gt[k]
	mat = spio.loadmat(gtFile, squeeze_me=True)
	M = mat['groundTruth']
	segg = M[0]['Segmentation'][()][:,:]
	bounn = M[0]['Boundaries'][()][:,:]
	big_struct = big_struct+[{'imgs':im1,'seggs':segg,'bounns':bounn}]

plt.imshow(segg)
plt.show()

