from flask import Flask, request, render_template, flash, redirect, url_for, send_file, Response
from werkzeug.utils import secure_filename
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

app = Flask(__name__)

prototxtPath= os.path.join("face_detector","deploy.prototxt")
weightsPath= os.path.join("face_detector","res10_300x300_ssd_iter_140000.caffemodel")
net= cv2.dnn.readNet(prototxtPath, weightsPath)
model= load_model('mask_detector.model')

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
        if confidence>0.5:
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

@app.route('/', methods=['GET'])
def main_page():
    return render_template('homepage.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        flash('No file part')
        return redirect(url_for('main_page'))
    imagefile = request.files['imagefile']
    filesavepath='online_detector_uploads/'+secure_filename(imagefile.filename)
    error=False
    if imagefile and allowed_file(imagefile.filename):
        filename = secure_filename(imagefile.filename)
        imagefile.save(filesavepath)
        image= cv2.imread(filesavepath)
        (h,w)= image.shape[:2]

        blob= cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))#normalization

        #computing face detection
        net.setInput(blob)
        detections= net.forward()
        for i in range(0,detections.shape[2]):
            confidence= detections[0,0,i,2]
            if confidence>0.5:
                box= detections[0,0,i,3:7]*np.array([w,h,w,h])
                (start_X,start_Y, end_X, end_Y)= box.astype('int')
                (start_X,start_Y)= (max(0,start_X),max(0,start_Y))
                (end_X,end_Y)= (min(w-1,end_X),min(h-1,end_Y))
                face= image[start_Y:end_Y, start_X:end_X]
                face= cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                face= cv2.resize(face, (224,224))
                face= img_to_array(face)
                face= preprocess_input(face)
                face= np.expand_dims(face, axis=0)

                (with_mask,without_mask)= model.predict(face)[0]
                if with_mask>without_mask:
                    label="Mask"
                    color= (0,255,0) 
                else:
                    label="No Mask"
                    color= (0,0,255)

                label="{}:{:.2f}%".format(label, max(with_mask,without_mask)*100) 
                cv2.putText(image,label,(start_X,start_Y-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
                cv2.rectangle(image, (start_X,start_Y),(end_X,end_Y),color,2)
        outputpath=os.path.join("static/online_detector_results",filename)
        cv2.imwrite(outputpath, image)
        return render_template('homepage.html', outputImage=filename,error=error)
    else:
        error=True
        return render_template('homepage.html', error=error)

@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    # Returning file from appended path
    return send_file( os.path.join('static/online_detector_results',filename), as_attachment=True)

@app.route('/streaming')
def streaming():
    return render_template('streaming.html')


#video streaming

video = cv2.VideoCapture(-1)

def gen_frames():
    while True:
        success, image = video.read()
        if not success:
            break
        else:
            (oheight, owidth) = image.shape[:2]
            adjusted_width = 400
            adjusted_height = int(adjusted_width * oheight / owidth)
            cv2.imshow("image", image)
            image = cv2.resize(image, (adjusted_width, adjusted_height), interpolation = cv2.INTER_AREA)
            (locs,preds)=detect_and_predict_mask(image,net,model)

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
                    cv2.putText(image,label,(start_X,start_Y-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
                    cv2.rectangle(image, (start_X,start_Y),(end_X,end_Y),color,2)

            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)