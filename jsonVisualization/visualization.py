import cv2
from tkinter import *
import numpy as np
import sys
import json

class Points():
    def __init__(self):
        self.frameNuber = -1
        self.xLeft = []
        self.xRight = []
        self.yLeft = []
        self.yRight = []
        self.cLeft = []
        self.cRight = []
        self.xBody = []
        self.yBody = []
        self.cBody = []
        self.minDist = sys.float_info.max
        # orientation = orientation between thumbs and other fingers
        self.lOrientation = []
        self.rOrientation = []
        self.frameDistance = 0  # distance from the current word to the next word

master = Tk()

canvas_width = 1920/2
canvas_height = 1080/2
w = Canvas(master,
           width=canvas_width,
           height=canvas_height)
w.pack()

wordsFrame = []
wordsFrame.append(Points())
ctrWordsFrame = 0

# with open("E:/yulia/openpose/african_american/african_american_000000000060_keypoints.json") as df:
with open("F:/[[MASTER]]/thesis/openpose/african_american/african_american_000000000060_keypoints.json") as df:
# with open("/home/yulia/openpose/african_american/african_american_000000000060_keypoints.json") as df:
    data = json.load(df)
    wordsFrame[ctrWordsFrame].frameNumber = ctrWordsFrame
    #each hand has 21 points. x,y,c --> 0 to 62
    for i in range(0,21):
        wordsFrame[ctrWordsFrame].xLeft.append((data['people'][0]['hand_left_keypoints_2d'][i*3])/2)
        wordsFrame[ctrWordsFrame].yLeft.append((data['people'][0]['hand_left_keypoints_2d'][i*3+1])/2)
        wordsFrame[ctrWordsFrame].cLeft.append(data['people'][0]['hand_left_keypoints_2d'][i*3+2])
        wordsFrame[ctrWordsFrame].xRight.append((data['people'][0]['hand_right_keypoints_2d'][i*3])/2)
        wordsFrame[ctrWordsFrame].yRight.append((data['people'][0]['hand_right_keypoints_2d'][i*3+1])/2)
        wordsFrame[ctrWordsFrame].cRight.append(data['people'][0]['hand_right_keypoints_2d'][i*3+2])

        x = wordsFrame[ctrWordsFrame].xLeft[i]
        y = wordsFrame[ctrWordsFrame].yLeft[i]
        w.create_oval(x-2, y-2, x+2, y+2, fill="yellow")
        x = wordsFrame[ctrWordsFrame].xRight[i]
        y = wordsFrame[ctrWordsFrame].yRight[i]
        w.create_oval(x - 2, y - 2, x + 2, y + 2, fill="yellow")

    for i in range(0,1):
        w.create_line(wordsFrame[i].xLeft[0],wordsFrame[i].yLeft[0],wordsFrame[i].xLeft[1],wordsFrame[i].yLeft[1], fill="red")
        w.create_line(wordsFrame[i].xLeft[0],wordsFrame[i].yLeft[0],wordsFrame[i].xLeft[5],wordsFrame[i].yLeft[5], fill="red")
        w.create_line(wordsFrame[i].xLeft[0],wordsFrame[i].yLeft[0],wordsFrame[i].xLeft[9],wordsFrame[i].yLeft[9], fill="red")
        w.create_line(wordsFrame[i].xLeft[0],wordsFrame[i].yLeft[0],wordsFrame[i].xLeft[13],wordsFrame[i].yLeft[13], fill="red")
        w.create_line(wordsFrame[i].xLeft[0],wordsFrame[i].yLeft[0],wordsFrame[i].xLeft[17],wordsFrame[i].yLeft[17], fill="red")

        w.create_line(wordsFrame[i].xRight[0],wordsFrame[i].yRight[0],wordsFrame[i].xRight[1],wordsFrame[i].yRight[1], fill="red")
        w.create_line(wordsFrame[i].xRight[0],wordsFrame[i].yRight[0],wordsFrame[i].xRight[5],wordsFrame[i].yRight[5], fill="red")
        w.create_line(wordsFrame[i].xRight[0],wordsFrame[i].yRight[0],wordsFrame[i].xRight[9],wordsFrame[i].yRight[9], fill="red")
        w.create_line(wordsFrame[i].xRight[0],wordsFrame[i].yRight[0],wordsFrame[i].xRight[13],wordsFrame[i].yRight[13], fill="red")
        w.create_line(wordsFrame[i].xRight[0],wordsFrame[i].yRight[0],wordsFrame[i].xRight[17],wordsFrame[i].yRight[17], fill="red")

    for i in range(0,20):
        if(i%4 != 0):
            w.create_line(wordsFrame[0].xLeft[i],wordsFrame[0].yLeft[i],wordsFrame[0].xLeft[i+1],wordsFrame[0].yLeft[i+1], fill="red")
            w.create_line(wordsFrame[0].xRight[i],wordsFrame[0].yRight[i],wordsFrame[0].xRight[i+1],wordsFrame[0].yRight[i+1], fill="red")

    for i in range(0, 25):
        wordsFrame[ctrWordsFrame].xBody.append((data['people'][0]['pose_keypoints_2d'][i*3])/2)
        wordsFrame[ctrWordsFrame].yBody.append((data['people'][0]['pose_keypoints_2d'][i*3+1])/2)
        wordsFrame[ctrWordsFrame].cBody.append(data['people'][0]['pose_keypoints_2d'][i*3+2])

        x = wordsFrame[ctrWordsFrame].xBody[i]
        y = wordsFrame[ctrWordsFrame].yBody[i]
        w.create_oval(x - 2, y - 2, x + 2, y + 2, fill="green")




mainloop()