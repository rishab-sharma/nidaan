import os
from inference import get_inference
from flask import Flask , render_template,request
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
import numpy as np

app = Flask(__name__, template_folder = 'templates', static_url_path = "/static")
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')  
def index():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT,'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        destintion = "/".join([target,'test.jpeg'])
        print(destintion)
        file.save(destintion)
        

    return render_template('complete.html')

@app.route('/predict',methods=['POST'])
def predict():
    img = image.load_img('images/test.jpeg',target_size=(512,512))
    im = image.img_to_array(img)
    im = preprocess_input(im)
    im = np.expand_dims(im,axis=0)
    prediction = get_inference(im)
    print(prediction)
    return (str(prediction))
  
if __name__=='__main__':
   app.run(host = '0.0.0.0', port = 5000,debug=True)         	
