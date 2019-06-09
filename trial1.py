import json
from pprint import pprint
import glob
import os
import re
import pandas as pd


col_names = ['idx','filename']
dfFileName = pd.DataFrame(columns=col_names)
col_names = ['jsonfileidx']
for i in range(0,21):
    col_names.append("p"+str(i)+"x")
    col_names.append("p"+str(i)+"y")
    col_names.append("p"+str(i)+"c")
dfLHand = pd.DataFrame(columns = col_names)
dfRHand = pd.DataFrame(columns = col_names)

col_names = ['jsonfileidx']
for i in range(0,25):
    col_names.append("p" + str(i) + "x")
    col_names.append("p" + str(i) + "y")
    col_names.append("p" + str(i) + "c")
dfBodyParts = pd.DataFrame(columns = col_names)

files = glob.glob("/home/yulia/openpose/outputTransition/*.json")
for f in files:
  fName = os.path.basename(f)
  number = re.findall('\d+',f)
  number = max(map(int,number))
  with open(f) as df:
   data = json.load(df)
  lHand = [number]
  rHand = [number]
  body = [number]

  for i in range(0,63):
    lHand.append(data['people'][0]['hand_left_keypoints_2d'][i])
    rHand.append(data['people'][0]['hand_right_keypoints_2d'][i])

  for i in range(0,75):
    body.append(data['people'][0]['pose_keypoints_2d'][i])

  dfFileName.loc[len(dfFileName)] = [number,fName]
  dfLHand.loc[len(dfLHand)] = lHand
  dfRHand.loc[len(dfRHand)] = rHand
  dfBodyParts.loc[len(dfBodyParts)] = body

with open("/home/yulia/openpose/african_american/african_american_000000000060_keypoints.json") as df:
    african_american_lastframe = json.load(df)
african_american_LHand = african_american_lastframe['people'][0]['hand_left_keypoints_2d']
african_american_RHand = african_american_lastframe['people'][0]['hand_right_keypoints_2d']
african_american_Body = african_american_lastframe['people'][0]['pose_keypoints_2d']

with open("/home/yulia/openpose/advisor/advisor_000000000000_keypoints.json") as df:
    advisor_firstframe = json.load(df)
advisor_LHand = advisor_firstframe['people'][0]['hand_left_keypoints_2d']
advisor_RHand = advisor_firstframe['people'][0]['hand_right_keypoints_2d']
advisor_Body = advisor_firstframe['people'][0]['pose_keypoints_2d']

african_american_LHand
