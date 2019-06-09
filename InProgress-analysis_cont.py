#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from pprint import pprint
import glob
import os
import re
import cv2
import pandas as pd
import numpy as np
import sys
from copy import deepcopy
from PIL import Image


# In[2]:


class Points(): 
    def __init__(self):
        self.frameNuber = -1
        self.index = -1
        self.xLeft = []
        self.xRight = []
        self.yLeft = []
        self.yRight = []
        self.cLeft = []
        self.cRight = []
        self.xBody = []
        self.yBody = []
        self.cBody = []
        self.xFace = []
        self.yFace = []
        self.xPalmLeft = 0
        self.xPalmRight = 0
        self.yPalmLeft = 0
        self.yPalmRight = 0
        self.minDist = sys.float_info.max
#orientation = orientation between thumbs and other fingers
        self.lOrientation = []
        self.rOrientation = []
        self.xLCenter = 0
        self.xRCenter = 0
        self.yLCenter = 0
        self.yRCenter = 0
        self.frameDistance = 0 #distance from the current word to the next word


# In[3]:


def LoadTransition():
    trans = []
#     point_ins = Points()
#     point_ins.xLeft = []
#     files = glob.glob("E:/yulia/openpose/transition3/*.json") 
    files = glob.glob("../jsons/transitions/*.json")
#     files = glob.glob("/home/yulia/openpose/output/*.json")
#     files = glob.glob("F:/[[][[]MASTER[]][]]/thesis/openpose/transition/*.json")
    ctrTrans = 0
    for f in files:
        fName = os.path.basename(f)
        number = re.split("(\d+)", fName)
        number = int(number[1])
        trans.append(Points())
        with open(f) as df:
            data = json.load(df)
            frameNumber = number
            trans[ctrTrans].frameNumber = number
            trans[ctrTrans].index = ctrTrans
            #each hand has 21 points. x,y,c --> 0 to 62
            trans[ctrTrans].xBody.append(data['people'][0]['pose_keypoints_2d'][4*3])
            trans[ctrTrans].xBody.append(data['people'][0]['pose_keypoints_2d'][7*3])
            trans[ctrTrans].yBody.append(data['people'][0]['pose_keypoints_2d'][4*3])
            trans[ctrTrans].yBody.append(data['people'][0]['pose_keypoints_2d'][7*3])
            
            trans[ctrTrans].xPalmLeft = (data['people'][0]['hand_left_keypoints_2d'][0])
            trans[ctrTrans].yPalmLeft = (data['people'][0]['hand_left_keypoints_2d'][1])
            trans[ctrTrans].xPalmRight = (data['people'][0]['hand_right_keypoints_2d'][0])
            trans[ctrTrans].yPalmRight = (data['people'][0]['hand_right_keypoints_2d'][1])
            
            xLeft = []; yLeft = []; xRight = []; yRight = []
            for j in range(4,21,4):
                xLeft.append(data['people'][0]['hand_left_keypoints_2d'][j*3])
                yLeft.append(data['people'][0]['hand_left_keypoints_2d'][j*3+1])
                xRight.append(data['people'][0]['hand_right_keypoints_2d'][j*3])
                yRight.append(data['people'][0]['hand_right_keypoints_2d'][j*3+1])
            for j in range(1,5):
                trans[ctrTrans].lOrientation.append(np.cross([xLeft[0] , yLeft[0]] ,[xLeft[j] , yLeft[j]]))
                trans[ctrTrans].rOrientation.append(np.cross([xRight[0], yRight[0]],[xRight[j],yRight[j]]))
            trans[ctrTrans].xLCenter = np.sum(xLeft)/5
            trans[ctrTrans].xRCenter = np.sum(xRight)/5
            trans[ctrTrans].yLCenter = np.sum(yLeft)/5
            trans[ctrTrans].yRCenter = np.sum(yRight)/5
        ctrTrans+=1
    return trans


# In[4]:


trans = LoadTransition()


# In[5]:


a=[9,5]
b=[3,1]
np.cross(b,a)


# In[6]:


input_text = str(input("enter a sentence"))
input_text = input_text.split(" ")


# In[7]:


# dir_path = "F:/[[MASTER]]/thesis/openpose/"
dir_path = "../jsons/"


# In[8]:


words = []
for i in range(len(input_text)):
    ctr = 0
    words.append([])
    for filename in os.listdir(dir_path+input_text[i]):
        number = int(re.split("(\d+)", os.path.basename(filename))[1])
        if (number < 3 or number >= len(os.listdir(dir_path+input_text[i]))-3):
            words[i].append(Points())
            with open(dir_path +input_text[i]+"/"+input_text[i]+"_"+ ((str)(number)).zfill(12)+"_keypoints.json") as df:
                data = json.load(df)
                xLeft = []; yLeft = []; xRight = []; yRight = []
                for j in range(4,21,4):
                    xLeft.append(data['people'][0]['hand_left_keypoints_2d'][j*3])
                    yLeft.append(data['people'][0]['hand_left_keypoints_2d'][j*3+1])
                    xRight.append(data['people'][0]['hand_right_keypoints_2d'][j*3])
                    yRight.append(data['people'][0]['hand_right_keypoints_2d'][j*3+1])
                for j in range(1,5):
                    words[i][ctr].lOrientation.append(np.cross([xLeft[0] , yLeft[0]] ,[xLeft[j] , yLeft[j]]))
                    words[i][ctr].rOrientation.append(np.cross([xRight[0], yRight[0]],[xRight[j],yRight[j]]))
                words[i][ctr].xBody.append(data['people'][0]['pose_keypoints_2d'][4*3])
                words[i][ctr].xBody.append(data['people'][0]['pose_keypoints_2d'][7*3])
                words[i][ctr].yBody.append(data['people'][0]['pose_keypoints_2d'][4*3])
                words[i][ctr].yBody.append(data['people'][0]['pose_keypoints_2d'][7*3])
                                 
                words[i][ctr].xPalmLeft = (data['people'][0]['hand_left_keypoints_2d'][0])
                words[i][ctr].yPalmLeft = (data['people'][0]['hand_left_keypoints_2d'][1])
                words[i][ctr].xPalmRight = (data['people'][0]['hand_right_keypoints_2d'][0])
                words[i][ctr].yPalmRight = (data['people'][0]['hand_right_keypoints_2d'][1])
                
                
                words[i][ctr].xLCenter = np.sum(xLeft)/5
                words[i][ctr].xRCenter = np.sum(xRight)/5
                words[i][ctr].yLCenter = np.sum(yLeft)/5
                words[i][ctr].yRCenter = np.sum(yRight)/5
                
                ctr+=1


# In[9]:


class Distance(object):
    def __init__(self):
        self.distance = 0
        self.From = -1
        self.index = -1


# In[29]:


def distance(framesA, framesB, framesC, weight):
    arr_distance = []
    frames = framesA+framesB
    
    min_x_L = min(frames[2].xLCenter, frames[3].xLCenter) 
    max_x_L = max(frames[2].xLCenter, frames[3].xLCenter)
    min_y_L = min(frames[2].yLCenter, frames[3].yLCenter)
    max_y_L = max(frames[2].yLCenter, frames[3].yLCenter)
    
    min_x_R = min(frames[2].xRCenter, frames[3].xRCenter)
    max_x_R = max(frames[2].xRCenter, frames[3].xRCenter)
    min_y_R = min(frames[2].yRCenter, frames[3].yRCenter)
    max_y_R = max(frames[2].yRCenter, frames[3].yRCenter)
    
    for i in range(len(framesC)):
        const = 0.7
        ctrConst = 0.1
        temp_distance = 0
        for j in range(len(frames)):
            wrist_distance = 0
            for k in range(len(framesC[i].xBody)):
                wristX = framesC[i].xBody[k]-frames[j].xBody[k]
                wristY = framesC[i].yBody[k]-frames[j].yBody[k]
                wrist_distance += np.sqrt(np.power(wristX,2)+np.power(wristY,2))
            
#             orientation = 320
            orientation = 0
            for k in range(0,4):
#                 if framesC[i].lOrientation[k] == frames[j].lOrientation[k]:
#                     orientation -= 40
#                 if framesC[i].rOrientation[k] == frames[j].rOrientation[k]:
#                     orientation -= 40
                if framesC[i].lOrientation[k] * frames[j].lOrientation[k] < 0:
                    orientation += 1
                if framesC[i].rOrientation[k] * frames[j].rOrientation[k] < 0:
                    orientation += 1
            palm_left =np.sqrt(np.power(framesC[i].xPalmLeft-frames[j].xPalmLeft,2)+np.power(framesC[i].yPalmLeft-frames[j].yPalmLeft,2))
            palm_right=np.sqrt(np.power(framesC[i].xPalmRight-frames[j].xPalmRight,2)+np.power(framesC[i].yPalmRight-frames[j].yPalmRight,2))
            palm_distance =  palm_left + palm_right
            if orientation >1:
                palm_distance += 200
            if j >= 3:
                ctrConst = -0.2
            const += ctrConst
            temp_distance += const * (palm_left+palm_right+palm_distance)
        arr_distance.append(Distance())
        arr_distance[i].From = framesC[i].frameNumber
        arr_distance[i].distance = (temp_distance/6)
        arr_distance[i].index = i
    arr_distance.sort(key = lambda x: x.distance, reverse = False)
    
#     print(np.mean(list(x.distance for x in arr_distance)))
#     print(np.std(list(x.distance for x in arr_distance)))
    for i in range(len(arr_distance)):
        if min_x_L-20 <= framesC[arr_distance[i].index].xLCenter <= max_x_L+20:
            if min_y_L-20 <= framesC[arr_distance[i].index].yLCenter <= max_y_L+20:
                if min_x_R-20 <= framesC[arr_distance[i].index].xRCenter <= max_x_R+20:
                    if min_y_R-20 <= framesC[arr_distance[i].index].yRCenter <= max_y_R+20:
                        return arr_distance[i]


# In[30]:


def trans_smoothing(frameA, frameB):
    new_img = cv2.addWeighted(frameA, 0.5, frameB, 0.5, 0)
    return new_img


# In[31]:


def divide_left(frameA, frameB, frameT, threshold):
    frame = distance(frameA, frameB, frameT, weight="left")
    if frame:
        if(frame.distance >= threshold):
            if threshold == 1000:
                threshold = frame.distance
            else:
                return
        new_frameB = deepcopy(frameB)
        new_frameB.insert(0,deepcopy(frameT[frame.index]))
        new_frameB.pop()
        temp.insert(0, frameT[frame.index])
        frameT.pop(frame.index)
        divide_left(frameA,new_frameB,frameT,threshold-30)


# In[32]:


def divide_right(frameA,frameB,frameT, threshold):
    frame = distance(frameA, frameB, frameT, weight="right")
    if frame:
        if(frame.distance >= threshold):
            if threshold == 1000:
                threshold = frame.distance
            else:
                return
        new_frameA = deepcopy(frameA)
        new_frameA.append(deepcopy(frameT[frame.index]))
        new_frameA.pop(0)
        temp.append(frameT[frame.index])
        frameT.pop(frame.index)
        divide_right(new_frameA, frameB, frameT, threshold-30)


# In[33]:


threshold = 1000
temp = []
transition  = []
for i in range(0, len(words)-1):
    divide_left([deepcopy(words[i][3]),deepcopy(words[i][4]),deepcopy(words[i][5])],[deepcopy(words[i+1][0]),deepcopy(words[i+1][1]),deepcopy(words[i+1][2])],deepcopy(trans), threshold)
    divide_right([deepcopy(words[i][3]),deepcopy(words[i][4]),deepcopy(words[i][5])],[deepcopy(words[i+1][0]),deepcopy(words[i+1][1]),deepcopy(words[i+1][2])],deepcopy(trans), threshold)
    transition.append(temp.copy())
    temp = []


# In[34]:


transition


# In[35]:


def readVideo(vid):
    cap = vid
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    
    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    return buf


# In[36]:


def readLargeVideo(vid, trans_idx):
    cap = vid
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((len(trans_idx), frameHeight, frameWidth, 3), np.dtype('uint8'))
    
    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, frame = cap.read()
        for i in range(len(trans_idx)):
            if fc == trans_idx[i].frameNumber:
                buf[i] = frame
        fc += 1
    cap.release()
    return buf


# In[37]:


#read transition video and save to array of images
videos = []
for i in range(len(input_text)):
#     videos.append(readVideo(cv2.VideoCapture('F:/[[MASTER]]/thesis/ASL/'+input_text[i]+'.mp4')))
    videos.append(readVideo(cv2.VideoCapture('../ASL/'+input_text[i]+'.mp4')))
# trans_img = readVideo(cv2.VideoCapture('F:/[[MASTER]]/thesis/ASL/transition.mp4'))
# trans_img = readVideo(cv2.VideoCapture('/home/yulia/Documents/ASL/transitions.mp4'))


# In[38]:


written_trans = []
for i in range(0, len(transition)):
    written_trans.append([])
    written_trans[i] = (readLargeVideo(cv2.VideoCapture('../ASL/transitions.mp4'),transition[i]))


# In[27]:


def check_optical_flow(frameA, frameB):
    prvs = cv2.cvtColor(frameA,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frameA)
    hsv[...,1] = 255 
    nxt = cv2.cvtColor(frameB,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2                             #direction
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) #magnitude
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    print(hsv[800][800])
#     cv2.imshow('frame2',bgr)
#     k = cv2.waitKey(0)
#     cv2.destroyAllWindows()


# In[39]:


output = []
for i in range(0, len(input_text)):
    for j in range(0, len(videos[i])):
        output.append(videos[i][j])
    if i < len(input_text)-1:
        prev = videos[i][j]
        for j in range(0, len(written_trans[i])):
            if j > 0:
                prev = written_trans[i][j-1]
#             check_optical_flow(prev, written_trans[i][j])
            output.append(trans_smoothing(prev, written_trans[i][j]))
            output.append(written_trans[i][j])


# In[40]:


height,width,channels = videos[0][0].shape
# out = cv2.VideoWriter('E:/yulia/output/trial-bener-1.mp4',cv2.VideoWriter_fourcc('F','M','P', '4'), 10, (width,height))
# out = cv2.VideoWriter('F:/[[MASTER]]/thesis/outputThesis/trial-amin-16.mp4',cv2.VideoWriter_fourcc('F','M','P', '4'), 20, (width,height))
out = cv2.VideoWriter('../outputThesis/trial-amin-16.mp4',cv2.VideoWriter_fourcc('F','M','P', '4'), 20, (width,height))
for i in range(0, len(output)):
        out.write(output[i])
out.release()


# In[23]:


written_trans[0][0].shape


# In[24]:


width


# In[ ]:





# In[ ]:





# In[ ]:




