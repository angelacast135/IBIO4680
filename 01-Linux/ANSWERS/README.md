# Lab01 - Introduction to Linux
The first assigment of computer vision course.


sipi_images: directory for answer question 3 and 4. Scripts inside directory 
Lab1: directory for answer 5 find_duplicates.sh. The script finds duplicated images in Lab1 folders.
BSR: Directory with the bsds500 dataset. Question from 6 to 10.

Question to be answered: 

3. How many users exist in the course server?
    In /home directory appears to exist 6 users: arayobi10, jamartinez1423, jc.leon, pa.arbelaez, rvandres and vision
    However, to see active users (the ones who are logged on) we can use the command w which displays info about the users currently on the machine and their processes. FROM: https://www.computerhope.com/unix/w.htm
  

4. What command will produce a table of Users and Shells sorted by shell (tip: using cut and sort)
    Using the command who to show every login session on the machine
    ```
    # Cut the output from who command delimmited by ' ' (space) and shows the 1 and 14 field (user and Shell), and sort nuumerically (-g).
    who | cut -d ' ' -f1,14 | sort -g
    ```

    FROM: https://www.computerhope.com/unix/ucut.htm
          https://www.computerhope.com/unix/usort.htm

5. Create a script for finding duplicate images based on their content (tip: hash or checksum) You may look in the internet for ideas, Do not forget to include the source of any code you use.
    ```
#!/bin/bash

# go to Home directory
cd ~/Chaos_Donkey/Lab01 

# remove the previous .txt created
rm duplicated_images.txt 2>/dev/null

# the following command finds the duplicated images in the folder Lab1 and saves the result in duplicated_images.txt. 
# the fisrt part of the code finds the not empty files and sorts them. Then it takes only the duplicated found files. First, the size is compared to finally compare the hash of the duplicated images in the current folder.
find -not -empty -type f -printf "%s\n" | sort -rn | uniq -d | xargs -I{} -n1 find -type f -size {}c -print0 | xargs -0 md5sum | sort | uniq -w32 --all-repeated=separate > duplicated_images.txt

chmod u+x duplicated_images.txt
    ```

    FROM: http://www.commandlinefu.com/commands/view/3555/find-duplicate-files-based-on-size-first-then-md5-hash

6. Download the bsds500 image segmentation database and decompress it (keep it in you hard drive, we will come back over this data in a few weeks).
    ```
    #For downloading the bsds500 dataset
    wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
    # Uncompress the .tgz file 
    tar -zxvf BSR_bsds500.tgz
    # Remove the .tgz file 
    rm BSR_bsds500.tgz
    ```

7. What is the disk size of the uncompressed dataset, How many images are in the directory 'BSR/BSDS500/data/images'?
    ```
    # Size of the uncompress dataset
    du -sh ~/Chaos_Donkey/Lab01/BSR
    # Output: 73M	/home/vision/Chaos_Donkey/Lab01/BSR

    # Count the number of images in BSR/BSDS500/data/images with extention .jpg   
    find ~/Chaos_Donkey/Lab01/BSR/BSDS500/data/images -name "*.jpg" -exec identify {} \; | wc -l
    ```
    Ouput: 500

8. What is their resolution, what is their format?
   The format of the images is jpg.
    ```   
   # find all files whose name end in .tif
   images=$(find ~/Chaos_Donkey/Lab01/BSR/BSDS500/data/images -name *.jpg)

    #iterate over them
    for im in ${images[*]}
    do
        # print the properties of each image. For greater detail you can add -verbose before the image. 
        # FROM: https://www.imagemagick.org/script/identify.ph
        identify $im
    done
    ```    

    The resolution of all images is: 321*481 = 154401, in both orientations (landscape: 481X321  and portrait: 321X481 )

9. How many of them are in landscape orientation (opposed to portrait)?
    ```
    find ~/Chaos_Donkey/Lab01/BSR/BSDS500/data/images -name "*.jpg" -exec identify {} \; | grep -i 481X321 | wc -l
    ```

    Output: 348 landscape images (Size = 481X321)

10. Crop all images to make them square (256x256). Tip: do not forget about imagemagick.
    
    ```
     # Move to dataset directory
    cd ~/Chaos_Donkey/Lab01/BSR/BSDS500/data/images

    # find all files whose name end in .tif
    images=$(find . -name *.jpg)

    #iterate over them
    for im in ${images[*]}
    do
        convert $im -gravity center -crop 260x260+0+0 $im 
    done
    ```

   FROM: https://www.imagemagick.org/discourse-server/viewtopic.php?t=13793
