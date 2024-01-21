import numpy as np
import cv2 as cv


haar_cascade=cv.CascadeClassifier('haar_face.xml')
people=['ajith','vijay']


face_recognizer=cv.face.LBPHFaceRecognizer_create()

face_recognizer.read('face_trained.yml')

img=cv.imread('./ajith/5.jpg')
cv.imshow("img",img)


gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)



face_rect=haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in face_rect:
    
    face_roi=gray[y:y+h,x:x+h]

    label,confidence=face_recognizer.predict(face_roi)
    
    print(f'Label={people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


cv.imshow("detect",img)

cv.waitKey(0)
