# Move to dataset directory
cd ~/Chaos_Donkey/Lab01/BSR/BSDS500/data/images

# find all files whose name end in .tif
images=$(find . -name *.jpg)

#iterate over them
for im in ${images[*]}
do
    convert $im -gravity center -crop 260x260+0+0 $im 
done
