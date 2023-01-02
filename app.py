import cv2
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request
import os
import uuid
import torch
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


IMG_SIZE = 299
categories = ["ArtDecor","Hitech","Indochina","Industrial","Scandinavian" ]
model = tf.keras.models.load_model(r"./xception_model_2.h5")


def save_crop_images(image): 
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # Inference
    results = model(image)
    model.cpu()  # CPU
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    model.max_det = 1000  # maximum number of detections per image
    model.amp = False  # Automatic Mixed Precision (AMP) inference
    crops = results.crop(save=False)

    predict_data = {} 
    predict_data=predict_crop_images(crops)
    return predict_data

def predict_crop_images(crops):
    predict_data = {}
    index = 0
    for img in crops:
                crop_img = img["im"]
                img_array = crop_img/255.0
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                new_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(new_array)
                #https://stackoverflow.com/q/43310681
                pil_img = Image.fromarray(crop_img)
                buff = BytesIO()
                pil_img.save(buff, format="JPEG")
                new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
                #
                styles = {}
                for ratio,style in zip(prediction[0],categories):
                    styles.update({style: '{0:.10f}'.format(ratio)})
                predict_data.update({str(index): {"crop_img": new_image_string, "predict":styles }})
                index = index+1
    return predict_data


    
def readBase64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def save_evaluate_data(receive_data):
    evaluate_img=(readBase64(receive_data["crops_img"]))
    save_dir = './evaluate_data'
    #for style in os.listdir(save_dir):
    cv2.imwrite(save_dir+"/"+receive_data["style"]+"/"+str(uuid.uuid4())+".jpg", evaluate_img)


@app.post("/detect")
def detect_upload_img():
    request_data = request.get_json()
    response =save_crop_images(readBase64(request_data['base64']))
    return response, 201

@app.post("/evaluation")
def receive_evaluation_img():
    receive_data = request.get_json()
    save_evaluate_data(receive_data)
    response="Success"
    return response, 201

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))