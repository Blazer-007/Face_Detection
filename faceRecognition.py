import numpy as np
import cv2
import os

#-------------------------------KNN STARTS-----------------------------------------

#First we define distance function
def dist(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())
    
#Now defining KNN Algorithm
def KNN(train,test,k=5):
    #X will have all the except last one as last one will contain the ID which is allocated to every image
    X=train[:,:-1]
    Y=train[:,-1]
    #creating a list which will save the value of distance between test and training point and will save id also.
    vals=[]
    
    for i in range(X.shape[0]):
        vals.append((dist(test,X[i]),Y[i]))
    
    """Now we will sort the vals on the basis of distance and will take only first 'k' values from it as they
       will be the nearest ones."""  
    vals=sorted(vals)
    vals=vals[:k]
    
    #Now we will find which class id is more closer to the test image.
    new_vals=np.unique(vals[1],return_counts=True)
    max_ind=np.argmax(new_vals[1])
    prediction=new_vals[0][max_ind]
    return(int(prediction))
    
#--------------------------------KNN ENDS----------------------------------------------------------

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = 'Data/'
labels = []

class_id = 0

names = {} #mapping b/w id and name

# Data Preparation
for fx in  os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)


face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))


print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

# Testing 

while True:
    ret , frame = cap.read()
    if ret == False :
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h = face

        #Get the ROI
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        #Predict
        out = KNN(trainset,face_section.flatten())

        #Display the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("Faces ",frame)

    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






