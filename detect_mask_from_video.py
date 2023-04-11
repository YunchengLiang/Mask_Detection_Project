from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import argparse
import numpy as np
import imutils
import time
import cv2
import os

parser=argparse.ArgumentParser()
parser.add_argument('-f','--face',default="face_detector", type=str,help="path to face detector model directory")
parser.add_argument('-m','--model',type=str,default='mask_detector.model',help="path to trained mask detector model")
parser.add_argument('-c','--confidence',type=float,default=0.5,help='probability threshold for detection')
args=parser.parse_args()

prototxtPath= os.path.join(args.face,"deploy.prototxt")
weightsPath= os.path.join(args.face,"res10_300x300_ssd_iter_140000.caffemodel")
net= cv2.dnn.readNet(prototxtPath, weightsPath)

print("load face mask detection model:")
mask_model= load_model(args.model)

def detect_and_predict_mask(frame,faceNet,maskModel):
    (h,w)= frame.shape[:2]
    blob= cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))#normalization
    faceNet.setInput(blob)
    detections= faceNet.forward()

    faces= []
    locs= []
    preds= []

    for i in range(0,detections.shape[2]):
        confidence= detections[0,0,i,2]
        if confidence>args.confidence:
            box= detections[0,0,i,3:7]*np.array([w,h,w,h])
            (start_X,start_Y, end_X, end_Y)= box.astype('int')
            (start_X,start_Y)= (max(0,start_X),max(0,start_Y))
            (end_X,end_Y)= (min(w-1,end_X),min(h-1,end_Y))
            face= frame[start_Y:end_Y, start_X:end_X]
            face= cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face= cv2.resize(face, (224,224))
            face= img_to_array(face)
            face= preprocess_input(face)
            faces.append(face)
            locs.append((start_X,start_Y,end_X,end_Y))
    
    if len(faces)>0:
        faces = np.array(faces, dtype="float32")
        preds= maskModel.predict(faces)
    
    return (locs,preds)

print("start streaming:")
stream=VideoStream(src=0).start()
time.sleep(5.0)

while True:
    frame= stream.read()
    frame= imutils.resize(frame, width=400)
    (locs,preds)=detect_and_predict_mask(frame,net,mask_model)

    for (loc,pred) in zip(locs,preds):
        (start_X,start_Y,end_X,end_Y)= loc
        (with_mask,without_mask)= pred
        if with_mask>without_mask:
            label="Mask"
            color= (0,255,0) 
        else:
            label="No Mask"
            color= (0,0,255)

        label="{}:{:.2f}%".format(label, max(with_mask,without_mask)*100) 
        cv2.putText(frame,label,(start_X,start_Y-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(frame, (start_X,start_Y),(end_X,end_Y),color,2)

    cv2.imshow('Frame',frame)
    key=cv2.waitKey(1)

    if key==ord('q'):
         break
    
cv2.destroyAllWindows()
stream.stop()


