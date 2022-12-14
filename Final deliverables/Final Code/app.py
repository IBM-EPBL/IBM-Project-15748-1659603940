#Importing Libraries
import os
import numpy as np
from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Routing to the HTML Page
app=Flask(__name__)
model=load_model('ECG.h5')

@app.route("/")
def about():
    return render_template("about.html")

@app.route("/about")
def home():
    return render_template("about.html")

@app.route("/info")
def information():
    return render_template("info.html")

@app.route("/upload")
def test():
    return render_template("index6.html")

@app.route("/predict",methods=["GET","POST"])
def upload():
    if request.method=='POST':
        f=request.files['file']
        basepath=os.path.dirname('__file__')
        filepath=os.path.join(basepath,"uploads",f.filename)
        f.save(filepath)

        img = image.load_img(filepath,target_size=(180,180))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        pred = model.predict(x)
        classes=np.argmax(pred,axis=1)
        print("prediction",pred)
        print(classes)

        index_list=['Left Bundle Branch Block','Normal','Premature Atrial Contraction',
        'Premature Ventricular Contraction','Right Bundle Branch Block','Ventricular Fibrillation']
        result = str(index_list[classes[0]])
        return render_template()
    return None

if __name__=="__main__":
    app.run(debug=False)
