import cv2
import numpy as np
import face_recognition

imgsuman = face_recognition.load_image_file('facerecog\suman\suman.JPG')
imgsuman = cv2.cvtColor(imgsuman,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('facerecog\suman\mahesh.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgsuman)[0]
encodesuman = face_recognition.face_encodings(imgsuman)[0]
cv2.rectangle(imgsuman,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


faceLoctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodesuman],encodetest)
faceDis = face_recognition.face_distance([encodesuman],encodetest)
print(results,faceDis)
cv2.putText(imgtest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('suman',imgsuman)
cv2.imshow(' test',imgtest)
cv2.waitKey(0)