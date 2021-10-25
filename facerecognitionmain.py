import cv2
import numpy as np
import  face_recognition

imageh= face_recognition.load_image_file('image/hs.jpg')
imageh= cv2.cvtColor(imageh, cv2.COLOR_BGR2RGB)

imagetest= face_recognition.load_image_file('image/lt.jpg')
imagetest= cv2.cvtColor(imagetest, cv2.COLOR_BGR2RGB)

facelocation= face_recognition.face_locations(imageh)[0]
encodeh= face_recognition.face_encodings(imageh)[0]
cv2.rectangle(imageh,(facelocation[3],facelocation[0]),(facelocation[1],facelocation[2]),(255,0,255),2)

facelocationtest= face_recognition.face_locations(imagetest)[0]
encodetest= face_recognition.face_encodings(imagetest)[0]
cv2.rectangle(imagetest,(facelocationtest[3],facelocationtest[0]),(facelocationtest[1],facelocationtest[2]),(255,0,255),2)

#print(facelocation)
result= face_recognition.compare_faces([encodeh],encodetest)
facedis= face_recognition.face_distance([encodeh],encodetest)
print(result, facedis)
cv2.putText(imagetest,f'{result}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('harry',imageh)
cv2.imshow('harry test',imagetest)
cv2.waitKey(0)
