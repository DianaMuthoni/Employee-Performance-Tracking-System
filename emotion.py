#!/usr/bin/env python


import cv2
import pyautogui
import numpy as np
import time


from deepface import DeepFace




face_cascade = cv2.CascadeClassifier(r'C:\Users\user\Desktop\My_Emotion\haarcascade_frontalface_default - Copy.xml')

TIMER = int(5)
k=0

starttime = time.time()


cap = cv2.VideoCapture(0)
######################
capture=int(1)


while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(img_path = frame , actions=['emotion'], enforce_detection=False )

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    emotion = result["dominant_emotion"]

    txt = str(emotion)

    cv2.putText(frame,txt,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.imshow('frame',frame)

    ################################

    #time.sleep(1)
    endtime = time.time()
    diff = endtime - starttime

    if diff > 5:
        starttime = time.time()

    if diff >= 5:
        k = k+1
        cv2.imwrite('C:/wamp/www/MAIN_PROJECT/My_Emotion/images/frame' + str(k) + '.jpg', frame)






#cv2.imwrite('C:/wamp/www/MAIN_PROJECT/My_Emotion/images/frame' + str(k) + '.jpg', frame)
    #cv2.imwrite('img.png', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()