#!/bin/bash

# Move to lab01 directory
cd ~/Chaos_Donkey/Lab01

# Download bsds500 image segmentation dataset and decompress it

wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
tar -zxvf BSR_bsds500.tgz
rm BSR_bsds500.tgz
