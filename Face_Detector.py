# Note
# Step 1:
# Step 2: Making them all back and white

import cv2
from random import randrange

# Load some pre-trained Data on face frontals form opencv (harr cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# img = cv2.imread('RDJ.jpg')
# img = cv2.imread('shreyash.jpg')
webcam = cv2.VideoCapture(0)


# Iterate forever over frame
while True:
    ## Read the current frames
    successful_frame_read, frame = webcam.read()

    # must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    for (x,y,w,h) in face_coordinates:
        # (x,y,w,h) = face_coordinates[0]
        cv2.rectangle(frame, (x, y), (x+w,y+h), (0,256,0,1) ,2)
    
    cv2.imshow('Shreyash Kakde Face Detector',frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key ==81 or key == 113:
        break
    
# Relese the VideoCapture Object
webcam.release()


# # Detect Faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)



# print(face_coordinates)



# # Draw Rectangle Around the face
# # the (77,160) refers to the left upper co-ordinate
# # the (346,346) refers to the right buttom co-ordinate 
# # the (0,255,0) refers to the BGR (blue,green,red) color
# # and the 2 refers to the thik ness of the rectangle
# for (x,y,w,h) in face_coordinates:
# # (x,y,w,h) = face_coordinates[0]
#     cv2.rectangle(img, (x, y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)),3) 




# # cv2.imshow('Shreyash Kakde Face Detector', img)
# # cv2.waitKey()
# cv2.imshow('Shreyash Kakde Face Detector', img)
# cv2.waitKey()

print("code completed")
