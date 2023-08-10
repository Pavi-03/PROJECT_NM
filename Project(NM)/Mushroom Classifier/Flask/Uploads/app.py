import numpy as np
import os
import tensorflow as tf
from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

model = load_model(r"mushroom.h5")
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/input')
def input1():
    return render_template("about.html")

@app.route('/input')
def input2():
    return render_template("images.html")
@app.route('/input')
def input3():
    return render_template("predict.html")
@app.route('/predict',methods=["GET","POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img = image.load_img(filepath,target_size=(224,224,3))
        x = image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data),axis=1)
        index = ['Boletus','Lactarius','Russula']
        result = str(index[prediction[0]])
        print(result)
        return render_template('classify.html',prediction = result)
    
if __name__ == "__main__":
    app.run(debug=False)