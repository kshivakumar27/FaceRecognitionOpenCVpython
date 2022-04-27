import cv2
import os
import numpy as np
import faceRecognition as fr

test_img=cv2.imread("test4.jpg")
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected :",faces_detected)

#for (x,y,w,h) in faces_detected:
    #cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,255),thickness=3)
    

#cv2.imshow("faces_detected ",resized_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows(0)


faces,faceID=fr.labels_for_training_data('trainingimages')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write("traininfData.yml")


name={0:"Thor",1:"Chandana"}#creating dictionary containing names for each label

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>37):#If confidence more than 37 then don't print predicted face text on screen
        continue
    fr.put_text(test_img,predicted_name,x,y)
resized_img=cv2.resize(test_img,(500,500))
cv2.imshow("face dtecetion ",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows
