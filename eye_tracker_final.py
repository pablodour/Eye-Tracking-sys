#!/usr/bin/env python
# coding: utf-8

# In[30]:


#imports
import cv2 as cv
import numpy as np
import time
import dlib


# In[31]:


#prediction models
facedetector = dlib.get_frontal_face_detector()
landmarkdetector = dlib.shape_predictor(
    "Predictor/shape_predictor_68_face_landmarks_GTX.dat")
#Álvarez Casado C. & Bordallo López M.  (2021) 
#Real-time face alignment: evaluation methods, 
#training strategies and implementation optimization 
#SPRINGER JOURNAL OF REAL-TIME IMAGE PROCESSING (in press)#

#variables
##gets ID of camera, delay for prediction in secs, 
##flag for what  eye to track, 0 right, 1 left, 2 both
def track_eyes(frame, delay, flag):
    
        #grayscaling the frame
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
        #getting the faces
        faces = facedetector(frame_gray)
    
        if faces:           
            PointList = landmarker(frame_gray, faces[0])

            RightEyePoint = PointList[36:42]
            LeftEyePoint = PointList[42:48]
        
            Rcropedeye = EyeCrop(frame_gray,RightEyePoint)
            Lcropedeye = EyeCrop(frame_gray, LeftEyePoint)

            speed = EyeTrack(Lcropedeye,Rcropedeye, flag)
            
        else:
            speed = 128
            
        time.sleep(delay)
        return speed


# In[32]:


#function declaration

def landmarker(frame, face):
    landmarks = landmarkdetector(frame, face)
    PointList = []
    # looping through each landmark
    for n in range(0, 68):
        point = (landmarks.part(n).x, landmarks.part(n).y)
        PointList.append(point)
    return PointList

def EyeCoordinates(eyePoints):
    maxX = (max(eyePoints, key=lambda item: item[0]))[0]
    minX = (min(eyePoints, key=lambda item: item[0]))[0]
    maxY = (max(eyePoints, key=lambda item: item[1]))[1]
    minY = (min(eyePoints, key=lambda item: item[1]))[1]
    
    return maxX, minX, maxY, minY

def EyeCrop(frame, Eye):
    
    ##creating mask for just taking the eyes
    framedim = frame.shape
    mask = np.zeros(framedim, dtype=np.uint8)
    eye = np.array(Eye, dtype=np.int32)
    cv.fillPoly(mask, [eye], 255)
    eyeImage = cv.bitwise_and(frame, frame, mask=mask)
    
    maxX, minX, maxY, minY = EyeCoordinates(Eye)
    
    bottom_left = (minX, minY)
    top_right = (maxX,maxY)
    
    eyeImage[mask == 0] = 255
    
    cropedeye = eyeImage[minY:maxY, minX:maxX]
    
    return cropedeye

def EyePixels(cropedEye):
    
    height, width = cropedEye.shape
    divPart = int(width/3)
    ret, thresholdEye = cv.threshold(cropedEye, 85, 255, cv.THRESH_BINARY_INV)
    rightPart = thresholdEye[0:height, 0:divPart]
    centerPart = thresholdEye[0:height, divPart:divPart+divPart]
    leftPart = thresholdEye[0:height, divPart+divPart:width]
    
    return rightPart, centerPart, leftPart
    
def EyeTrack(LeftEye, RightEye, flag):
    
    #flag will determine which eye is considered
    
    LrightPart, LcenterPart, LleftPart = EyePixels(LeftEye)
    RrightPart, RcenterPart, RleftPart = EyePixels(RightEye)
    if flag == 0: #right eye only
        rightBlackPx = np.sum([RrightPart == 255])
        centerBlackPx = np.sum([RcenterPart == 255])
        leftBlackPx = np.sum([RleftPart == 255])
        AllPx = np.sum([RightEye==255])
    elif flag== 1: #left eye only
        rightBlackPx = np.sum([LrightPart == 255])
        centerBlackPx = np.sum([LcenterPart == 255])
        leftBlackPx = np.sum([LleftPart == 255])
        AllPx = np.sum([LeftEye==255])
    else: #both eyes
        rightBlackPx = np.sum([LrightPart == 255])
        rightBlackPx += np.sum([RrightPart == 255])
        centerBlackPx = np.sum([LcenterPart == 255])
        centerBlackPx += np.sum([RcenterPart == 255])
        leftBlackPx = np.sum([LleftPart == 255])
        leftBlackPx += np.sum([RleftPart == 255])
        AllPx = np.sum([RightEye==255])
        AllPx += np.sum([LeftEye==255])
    if(abs(rightBlackPx-leftBlackPx)>30):
        idx = [rightBlackPx, leftBlackPx].index(max([rightBlackPx, leftBlackPx]))
        if idx==0 : ##right
            thres = rightBlackPx/AllPx
        else: ##left
            thres = -leftBlackPx/AllPx
    else: ##center
        thres = 0
            
    
    speed = 128*(1+thres)
    return speed

