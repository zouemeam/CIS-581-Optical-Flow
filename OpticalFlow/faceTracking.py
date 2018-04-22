'''
  File name: faceTracking.py
  Author:
  Date created:
'''

'''
  File clarification:
    Generate a video with tracking features and bounding box for face regions
    - Input rawVideo: the video contains one or more faces
    - Output trackedVideo: the generated video with tracked features and bounding box for face regions
'''

def faceTracking(rawVideo):
  #TODO: Your code here
  import numpy as np 
  import cv2 
  import scipy 
  import matplotlib.pyplot as plt
  import matplotlib
  from detectFace import detectFace
  from getFeatures import getFeatures
  from estimateAllTranslation import estimateAllTranslation
  from applyGeometricTransformation import applyGeometricTransformation
  frameSet=[]#list to store all frames of the input video
  newFrameSet=[]#list to store all frames of the output video
  tf= True 
  plt.ioff()
  while tf:#read each frame in the input video
     tf,frame=rawVideo.read()
     frameSet.append(frame)
  frameSet=frameSet[:-1]

  bbox=detectFace(frameSet[0])#detect bounding box in the first frame 
  gray=cv2.cvtColor(frameSet[0],cv2.COLOR_BGR2GRAY)#convert first frame to gray scale image
  x,y=getFeatures(gray,bbox)#extract feature points from gray scale image and bounding box
  
  #drawing
  plt.imshow(cv2.cvtColor(frameSet[0], cv2.COLOR_BGR2RGB))
  [r1b,c1b,d1b]=np.asarray(bbox.shape)
  for i in range(r1b):#plot bounding box and feature points 
      b=bbox[i,:,:]
      xloc=x[:,i]
      yloc=y[:,i]
      facebb=matplotlib.patches.Polygon(b,closed=True,fill=False)
#      facebb.set_edgecolor('w')#uncomment if bounding box color blends with black background
      features=plt.plot(xloc,yloc,'w.',ms=1)
      plt.gca().add_patch(facebb)
  plt.axis('off')
  plt.savefig("temp.png",dpi=300,bbox_inches="tight")
  img=cv2.imread("temp.png")
  plt.close()
  newFrameSet.append(img);

#getting features and transforming 
  for k in range(1,len(frameSet)):#iterate through all frames of the input video 
      newXs,newYs =estimateAllTranslation(x,y,frameSet[k-1],frameSet[k])
      Xs,Ys,newbbox=applyGeometricTransformation(x,y,newXs,newYs,bbox)
      plt.imshow(cv2.cvtColor(frameSet[k], cv2.COLOR_BGR2RGB))
      print len(Xs)
      for j in range (r1b):
          b=newbbox[j,:,:]
          xloc=Xs[:,j]
          yloc=Ys[:,j]
          facebb=matplotlib.patches.Polygon(b,closed=True,fill=False)
#          facebb.set_edgecolor('w')#uncomment if bounding box color blends with black background
          features=plt.plot(xloc,yloc,'w.',ms=1)
          plt.gca().add_patch(facebb)
      plt.axis('off')
      plt.savefig("temp.png",dpi=300,bbox_inches="tight")
      img=cv2.imread("temp.png")
      plt.close()
      newFrameSet.append(img);
      x=Xs
      y=Ys
      bbox=newbbox
    
  [height,width,layer]=np.asarray(newFrameSet[0].shape)
  trackedVideo=cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'MP4V'),30,(width,height))#write frames to a video in mp4 format and 30fps
  for m in range (len(newFrameSet)):
      trackedVideo.write(newFrameSet[m].astype('uint8'))
      cv2.destroyAllWindows()
  trackedVideo.release()
  
  return trackedVideo