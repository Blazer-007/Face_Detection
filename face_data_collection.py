import cv2
import numpy as np

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = 'Data/'

file_name = input('Enter name of person : ')

while True:

    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret==False:
        continue

    
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3])

    # Pick the largest face (last face)


    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        #extract (crop out the required face ) : Region of Interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        skip += 1
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))
            cv2.imshow("Face Section",face_section)


    cv2.imshow("Frame",frame)
    

    if(skip%10==0):
        pass





    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0]),-1)
print(face_data.shape)

np.save(dataset_path+file_name+'.npy',face_data)
print("data successfully saved at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()