'''
  File name: applyGeometricTransformation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for bounding box
    - Input startXs: the x coordinates for all features wrt the first frame
    - Input startYs: the y coordinates for all features wrt the first frame
    - Input newXs: the x coordinates for all features wrt the second frame
    - Input newYs: the y coordinates for all features wrt the second frame
    - Input bbox: corner coordiantes of all detected bounding boxes
    
    - Output Xs: the x coordinates(after eliminating outliers) for all features wrt the second frame
    - Output Ys: the y coordinates(after eliminating outliers) for all features wrt the second frame
    - Output newbbox: corner coordiantes of all detected bounding boxes after transformation
'''

def applyGeometricTransformation(startX, startY, newXs, newYs, bbox):
  #TODO: Your code here
  import numpy as np 
  import skimage.transform
  
  
  [row,col]=np.asarray(startX.shape)#determine the dimensions of feature points
  [rb,cb,db]=np.asarray(bbox.shape)#find the number of detected faces
  newbbox=np.zeros([rb,cb,db])#preallocate return variable
  xlist=[]#initialize empty list to store x location of feature points
  ylist=[]#initialize empty list to store y location of feature points 
  numPt=0#keep track of max number of feature points for each face 
  for i in range(col):#loop through feature points for each face
      initX=startX[:,i]
      initY=startY[:,i]
      initX=initX[initX!=-1]
      initY=initY[initY!=-1]#filter out fillers of starting coordinates 
      newX=newXs[:,i]
      newY=newYs[:,i]
      newX=newX[newX!=-1]
      newY=newY[newY!=-1]#filter out fillers of new coordinates
      
      oldPos=np.matrix.transpose(np.vstack((initX,initY)))#reshape x and y coordinates to form a 2xn matrix
      newPos=np.matrix.transpose(np.vstack((newX,newY)))#
      
      t = skimage.transform.SimilarityTransform()#initialize a similarity transform object 
      t.estimate(oldPos, newPos)#estimate transformation matrix by passing in new and old coordinates
      
      coord=np.vstack((initX,initY,np.ones(len(newY))))#reshape x and y coordinates to form a 3xn matrix
      newCoord=np.dot(t.params,coord)#compute new coordinates using the transformation matrix
      diff=np.vstack((newX,newY))-newCoord[0:2,:]#compute difference between input new coordinates and computed new coordinates
      diff=diff*diff#find the square of the difference
      dist=diff[0,:]+diff[1,:]#find the sum of squares
      
      newX=newX[dist<=16]#filter out any feature points that deviated from the computed values by more than 4 pixels 
      newY=newY[dist<=16]
      initX=initX[dist<=16]
      initY=initY[dist<=16]
      t.estimate(np.matrix.transpose(np.vstack((initX,initY))),np.matrix.transpose(np.vstack((newX,newY))))
      #re-estimate tranformation matrix using filtered feature points 
      if len(newX)>numPt:
          numPt=len(newX)#update max number of feature points 
      xlist.append(newX)#add filtered feature points to output list
      ylist.append(newY)#
      boxcoord=np.vstack((np.matrix.transpose(bbox[i,:,:]),np.ones(4)))
      newBoxCoord=np.dot(t.params,boxcoord)#compute the transformation of bounding box 
      newbbox[i,:,:]=np.matrix.transpose(newBoxCoord[0:2,:])#allocate newbbox return variable 
  Xs=np.ones([numPt,col])*-1#preallocate Xs return variable
  Ys=np.ones([numPt,col])*-1#preallocate Ys return variable 
  for k in range(col):
      Xs[0:len(xlist[k]),k]=xlist[k]#allocate Xs and Ys return variables from xlist and ylist 
      Ys[0:len(ylist[k]),k]=ylist[k]#
  return Xs,Ys,newbbox