#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import airsim

import pprint
import os
import time
import numpy as np
from PIL import Image

pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()
client.confirmConnection()

def airsimSaveImage(location):
    """
    Get an image from the airsim API, saving it at the provided location. 
    Location should be a filename ending in .png
    """
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])

    for i, resp in enumerate(responses):
        im = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img_rgb = im.reshape(resp.height, resp.width, 4)
        img = Image.fromarray(img_rgb, mode="RGBA").convert('LA')
        img.save(os.path.normpath(location))

airsimSaveImage("hello.png")

# currently reset() doesn't work in CV mode. Below is the workaround
#client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)