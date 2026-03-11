import os
import io
from PIL import Image
from fastapi import FastAPI
from flask import Flask,render_template,request
import numpy as np
app = Flask(__name__,template_folder = 'templates')
from call_resnet import classify
from callyolo import callyolo
@app.route('/')
def index():
    return render_template('index.html') 
@app.route('/upload',methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']
    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = np.array(img)
        return f"Classification : {callyolo(img)}"
if __name__ == '__main__':   app.run(debug=True)