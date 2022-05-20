from re import X
from flask.wrappers import Response
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import shutil
from PIL import Image
import torch.nn.functional as F
from flask import Flask,request,render_template,send_file,jsonify
import json
import io
import cv2
from inference import Inferencer
from gevent import pywsgi

app = Flask(__name__)
inferencer = Inferencer('checkpoint\car.model.epoch.44', classes=242)

@app.route('/classify', methods=['POST'])
def classify():
    img = request.files.get('file')
    path = os.path.join('data/', 'demo.jpg')
    img.save(path)
    
    img = Image.open(path)
    result = {'pred': inferencer(img)}
    
    return jsonify(result)

if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    server = pywsgi.WSGIServer(('127.0.0.1', 12345), app)
    server.serve_forever()