#!/bin/bash

# go to Home directory
cd ~/Chaos_Donkey/Lab01 # or just cd

# remove the folder created by a previous run from the script
rm -r duplicated_images 2>/dev/null

# create output directory
mkdir duplicated_images

# find all files whose name end in .tif
images=$(find sipi_images -name *.tiff)

for im1 in ${images[*]}
do
    for im2 in ${images[*]}
    do
    if [$im1!=$im2]
    then
        if [ cksum im1 | cut -f1 -d ' '==cksum im2 | cut -f1 -d ' ']
        then
            echo $im1 is duplicated
            cp  $im1 ~/Chaos_Donkey/Lab01/duplicated_images
        fi
    else
        echo $im1 and $im2 are the same
    fi
    done
done
