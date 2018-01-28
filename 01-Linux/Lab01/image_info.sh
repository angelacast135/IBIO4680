#!/bin/bash

 # find all files whose name end in .tif
    images=$(find ~/Chaos_Donkey/Lab01/BSR/BSDS500/data/images -name *.jpg)

    #iterate over them
    for im in ${images[*]}
    do
        # print the properties of each image. For greater detail you can add -verbose before the image.
        # FROM: https://www.imagemagick.org/script/identify.php
        identify $im
    done

