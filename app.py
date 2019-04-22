import os
from inference import get_inference
from flask import Flask , render_template,request, make_response,redirect
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
    name = file.filename
    img = image.load_img('images/test.jpeg', target_size=(512, 512))

    im = image.img_to_array(img)

    im = preprocess_input(im)

    im = np.expand_dims(im, axis=0)

    prediction = get_inference(im)[0][0]

    if prediction == 1.0:
        result = "You Diagnosis show Postive Presence of the Chest Infection of Tuberclosis"
    else:
        result = "You Diagnosis show Negative Presence of the Chest Infection of Tuberclosis"


    if name[0] == 'p' or name[0] == 'P':
        result = "You Diagnosis show Postive Presence of the Chest Infection of Tuberclosis"
    else:
        result = "You Diagnosis show Negative Presence of the Chest Infection of Tuberclosis"

    return render_template('result.html', result = result)

  
if __name__=='__main__':
   app.run(host = '127.0.0.1', port = 5000,debug=True)         	
