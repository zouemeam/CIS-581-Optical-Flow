# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:37:24 2017

@author: zoue
"""

import cv2 
from faceTracking import faceTracking 

video=cv2.VideoCapture('TyrionLannister.mp4')#change video name to run another video
out=faceTracking(video)#returns cv2.video object, actual output video in mp4 format is stored as output.mp4 in same folder


    
