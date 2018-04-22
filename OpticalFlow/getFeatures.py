'''
  File name: getFeatures.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detect features within each detected bounding box
    - Input img: the first frame (in the grayscale) of video
    - Input bbox: the four corners of bounding boxes
    - Output x: the x coordinates of features
    - Output y: the y coordinates of features
'''

def getFeatures(img, bbox):
  #TODO: Your code here
    import numpy as np
    import cv2
    import scipy
    from skimage.feature import corner_shi_tomasi,corner_harris
    [row,col,dim]=np.asarray(bbox.shape)#find the number of faces detected
    bbox=bbox.astype(int)#cast corners of the bounding box to integer
    neighbors=np.ones((3,3))#define mask for neighbors of local maximum suppression
    xcoord=[]#preallocate feature x coordinates list
    ycoord=[]#preallocate feature y coordinates list
    numPt=0#keep track of max number of feature points for each face
    for i in range(row):
        currentFace=img[bbox[i,0,1]:bbox[i,2,1],bbox[i,0,0]:bbox[i,1,0]]#extract image that is in the bounding box
        currentFace=corner_harris(currentFace)#corner edge detection
        idx=(currentFace>0.0005)#threshold suppression 
        currentFace=currentFace*idx#filter out points that did not make it past threshold suppression
        localM=scipy.ndimage.filters.maximum_filter(currentFace,footprint=neighbors)#local maximum suppression using 3x3 neighborhood
        msk=(currentFace==localM)#
        msk=msk*currentFace#
        [y,x]=np.where(msk>0)#
        x=x+bbox[i,0,0]#shift coordinates back to their real position
        y=y+bbox[i,0,1]
        if len(x)>numPt:
            numPt=len(x)#update max feature points 
        xcoord.append(x)#add detected features to their respective list 
        ycoord.append(y)
    x=np.ones((numPt,row))*-1#preallocate x coordinate return variable 
    y=np.ones((numPt,row))*-1#preallocate y coordinate return variable 
    for k in range(row):#fill in return variables based on contents xcoord and ycoord lists
        x[0:len(xcoord[k]),k]=xcoord[k]
        y[0:len(ycoord[k]),k]=ycoord[k]
    return x,y


