'''
  File name: detectFace.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detect or hand-label bounding box for all face regions
    - Input img: the first frame of video
    - Output bbox: the four corners of bounding boxes for all detected faces
'''

def detectFace(img):
  #TODO: Your code here
  import numpy as np
  import cv2 
  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#convert image to gray scale
  face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#create facial classifier 
  face=face_cascade.detectMultiScale(gray,1.2,5)#detect faces using the classifier 
  bbox=np.zeros((len(face),4,2))#preallocate variable to store corners of the bounding box
  for i in range(len(face)):#convert result of facial detection to corners of bounding box
     bbox[i,:,0]=np.array([face[i,0],face[i,0]+face[i,2],face[i,0]+face[i,2],face[i,0]])
     bbox[i,:,1]=np.array([face[i,1],face[i,1],face[i,1]+face[i,3],face[i,1]+face[i,3]])
  
  return bbox