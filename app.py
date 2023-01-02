from flask import Flask, request,jsonify
from flask_cors import CORS
import os
import uuid
app = Flask(__name__)
CORS(app)

from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

import base64
import cv2
import numpy as np
from main import save_crop_images,predict_crop_images
import json 
def readBase64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img
def save_evaluate_data(receive_data):
    evaluate_img=(readBase64(receive_data["crops_img"]))
    save_dir = './evaluate_data'
    count=0
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