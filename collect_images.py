# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 05:24:10 2019

@author: chilton

This code collects images from the airsim simulator and stores them to disk.
"""

import airsim

import pprint
import os
from pathlib import Path
import numpy as np
import time
from PIL import Image


def SimCoordinates(pathPoints, delta):
    """
    SimCoordinates produces a list of points along the path where images will
    be collected (one image at each point). We will refer to these points as
    image points.
    
    Input:
            pts -- array of points, (x,y), that define the path. Each point is
            a node in the path. The nodes are connected by straight lines. We
            will refer to these points as path points.
            delta -- the nominal distance between image points
    
    Output:
            numpy array of image points (x,y)
    """
# m is the number of stright line segments in the path
    m = len(pathPoints) - 1
# lp is the list/array of image points
    imagePoints = np.array([pathPoints[0]])
# loop over each path segment
    for i in range(m):
# L is the length of the current path segment
        L = np.sqrt((pathPoints[i + 1][0] - pathPoints[i][0])**2 + 
                    (pathPoints[i + 1][1] - pathPoints[i][1])**2)
# ni is the number of images to be taken on the current path segment
        ni = int(np.ceil(L/delta))
# dt is the increment in the path parameter between images
        dt = 1 / ni
# loop over each image position
        for j in range(ni):
# nx, ny are the x,y coordinates of the image point
            nx = pathPoints[i][0] + (pathPoints[i+1][0] -
                           pathPoints[i][0]) * dt * (j + 1)
            ny = pathPoints[i][1] + (pathPoints[i+1][1] -
                           pathPoints[i][1]) * dt * (j + 1)
            imagePoints = np.concatenate((imagePoints, [[nx, ny]]), axis=0)
    return imagePoints

# todo: remove for loop
def airsimSaveImage(x, y, z, filename):
    """
    Get an image from the airsim API, saving it at the provided filename. 
    filename should be a filename ending in .png
    """
# position the camera in x, y, z space
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0)), True)

    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    im = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = im.reshape(response.height, response.width, 3)
    img = Image.fromarray(img_rgb, mode="RGB").convert('LA')
    img.save(os.path.normpath(filename))
    time.sleep(0.5)


pp = pprint.PrettyPrinter(indent=4)

# this is a path, delta combination that will produce close to 750 images
#pathPoints = np.array([[-200, -200], [-200, 200], [0, 200], [0, -200], [200, -200], [200, 200]])
pathPoints = np.array([[-200, -200], [-200, 200], [0, 200], [0, -200]])
delta = 40
imagePoints = SimCoordinates(pathPoints, delta)
print("number of image points = ", imagePoints.size)
print("image points = ", imagePoints)

client = airsim.VehicleClient()
client.confirmConnection()
# point camera straight down, -pi/2 radians
client.simSetCameraOrientation(0, airsim.to_quaternion(-1.5708, 0, 0))

camera_info = client.simGetCameraInfo(str(0))
#print("CameraInfo: %s" % (pp.pprint(camera_info)))

# idir is the director where the images will be stored
#idir = r'C:\\Users\\chilton\\Dropbox\\a_Research_Active\\AFIT\\Python_Airsim\\images\\'
idir = Path("C:\\Users\\chilton\\Dropbox\\a_Research_Active\\AFIT\\Python_Airsim\\images\\")
print ("Saving images to %s" % idir)

zvals = np.linspace(-60, -70, num=2, endpoint=True)


for j in np.arange(zvals.size):
    tdir = Path(idir, str(np.abs(zvals[j])))
    tdir.mkdir(exist_ok=True, parents=True)
    for i in np.arange(imagePoints.shape[0]):
#    for i in np.arange(30):
        filename = tdir / ('image_' + str(i) + '.png')
        #filename = os.path.join(idir, 'image_' + str(i) + '.png')
        print("\ni = %d, filename = %s" % (i, filename))
        print("\nposition = %f %f %f" % (imagePoints[i,0], imagePoints[i,1], zvals[j]))
        airsimSaveImage(imagePoints[i,0], imagePoints[i,1], zvals[j], filename)

# currently reset() doesn't work in CV mode. Below is the workaround
#client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)









