"""
Author: Sean Pinder

This script takes all of the training images and reduces their bit depth from 32 to 8.
"""
import os
from PIL import Image

# Change your directory to the directory that has the data in it
# cd = 'C:/Users/Sean study/Documents/SeniorProject/Data'
cd = 'C:/Users/Sean study/Documents/SeniorProject/Data'
os.chdir(cd)

# The names of the files in your current directory
md_filenames = [str(i)[11:-2] for i in list(os.scandir())]

# Loops over each subdirectory, changing your current directory each time
for i in md_filenames:
    os.chdir(cd + '/' + i)
    sd_filenames = [str(i)[11:-2] for i in list(os.scandir())]
    # Loops over each image, reduces its bit depth, and then saves it 
    for i in sd_filenames:
        im = Image.open('image_0.png')
        a = im.convert('L')
        a.save('image_0.png')