from keras.layers import *
from keras.models import *
import numpy as np
import keras.backend as K
def get_inference(img):
    K.clear_session()

    model = Sequential()
    model.add(Conv2D(64, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding='valid',input_shape = (512,512,3),      name='block1_conv1'))
    model.add(Conv2D(64, kernel_size = (3,3), strides = (1,1), activation = 'relu',padding='valid', name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2),padding='valid',name='block1_pool'))

    model.add(Conv2D(128, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding='valid', name='block2_conv1'))
    model.add(Conv2D(128, kernel_size = (3,3), strides = (1,1), activation = 'relu',padding='valid', name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2),padding='valid',name='block2_pool'))

    model.add(Conv2D(256, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding='valid', name='block3_conv1'))
    model.add(Conv2D(256, kernel_size = (3,3), strides = (1,1), activation = 'relu',padding='valid', name='block3_conv2'))
    model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2),padding='valid',name='block3_pool'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1028, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.load_weights('Pneumonia_model.h5')
    prediction = model.predict(img)
    return prediction
    
if __name__=='__main__':
   img = image.load_img('test.jpeg',target_size=(512,512))
   im = image.img_to_array(img)
   im = preprocess_input(im)
   im = np.expand_dims(im,axis=0)
   prediction = get_inference(im)
   print(prediction)
   
