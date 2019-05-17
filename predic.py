import os
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.models import Sequential,load_model

#initialize model paths and model weights
img_width,img_height=32,32
model_path="./models/model.h5"
model_weight_path="./models/weights.h5"

#load model path and weight
model=load_model(model_path)
model.load_weights(model_weight_path)

#load image and convert to its target size
x=load_img('./sample/cat_3.png',target_size=(img_width,img_height))

#load image to an array
x=img_to_array(x)
x=np.expand_dims(x,axis=0)

#predict the image from model
array=model.predict(x)
print(array)
