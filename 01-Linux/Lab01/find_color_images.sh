#!/bin/bash

# Go to lab directory
cd ~/Chaos_Donkey/Lab01

#remove the folder created by a previous run from the script
rm -r color_images 2>/dev/null

# create output directory
mkdir color_images

#find all files whose name end in .tiff
images=$(find sipi_images -name *.tiff)

#iterate over them
for im in ${images[*]}
do
    # check if the output from identify contains the word "gray"
    identify $im | grep -q -i gray

    # $? give the exit code of the last command, in this case frep, it will be zero if a match was found
    if [ $? -eq 0 ]
    then
        echo $im is gray
    else 
        echo $im is color 
        cp $im ~/Chaos_Donkey/Lab01/color_images
    fi
done
