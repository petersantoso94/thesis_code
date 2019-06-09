import mysql.connector
import json
from pprint import pprint
import glob
import os
import re

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="yulia",
  database="posejson"
)

mycursor = mydb.cursor()

# sql = """CREATE TABLE bodypart(
#         JFK_idx BIGINT(20),"""
# for i in range(0, 25):
#   sql += "p" + str(i) + "x float(7,3), "
#   sql += "p" + str(i) + "y float(7,3), "
#   sql += "p" + str(i) + "c float(7,6)"
#   if i == 24:
#     break
#   sql += ", "
# sql += ")"

# print(sql)
# try:
#   #execute sql command
#   mycursor.execute(sql)
#   mydb.commit()
# except:
#   #rollback
#   mydb.rollback()
#
# mydb.close()



files = glob.glob("/home/yulia/openpose/outputTransition/*.json")
for f in files:
  fName = os.path.basename(f)
  number = re.findall('\d+',f)
  number = max(map(int,number))
  with open(f) as df:
   data = json.load(df)
  lHand = data['people'][0]['hand_left_keypoints_2d']
  rHand = data['people'][0]['hand_right_keypoints_2d']
  body = data['people'][0]['pose_keypoints_2d']
  # print(body)
  sqlFName = "INSERT INTO jsonfile values("+str(number)+",'"+fName+"')"
  sqlLHand = "INSERT INTO lefthand values("+str(number)+","
  sqlRHand = "INSERT INTO righthand values("+str(number)+","
  sqlBodyPose = "INSERT INTO bodypart values("+str(number)+","
  for i in range(0,63):
    sqlLHand += str(lHand[i])
    sqlRHand += str(rHand[i])
    if(i == 62):
      break
    sqlLHand += ","
    sqlRHand += ","
  sqlLHand +=")"
  sqlRHand += ")"
  for i in range(0,75):
    sqlBodyPose += str(body[i])
    if(i == 74):
      break
    sqlBodyPose+=","
  sqlBodyPose += ")"
    #execute sql command
  mycursor.execute(sqlFName)
  mycursor.execute(sqlLHand)
  mycursor.execute(sqlRHand)
  mycursor.execute(sqlBodyPose)
  mydb.commit()

mydb.close()
