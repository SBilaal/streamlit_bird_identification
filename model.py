from tensorflow.keras.preprocessing import image
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image, ImageOps
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import numpy as np
import os


classes = ['AFRICAN FIREFINCH', 'CROWNED PIGEON', 'GREEN JAY', 'MOURNING DOVE', 'NICOBAR PIGEON', 'PURPLE FINCH', 'RED BROWED FINCH', 'ROCK DOVE', 'STRAWBERRY FINCH', 'YELLOW HEADED BLACKBIRD']

root = 'D:\\MTE 500\Project\\Birds_Test_Dataset'
model_path1 = f'{root}\\app\\trial7_model.h5'
print(model_path1)

def build_model():
  xception = keras.applications.xception.Xception(include_top=False)

  input = keras.Input(shape=(299,299,3))
  x = xception(input, training=False)
  x = GlobalAveragePooling2D()(x)
  output = Dense(units=10, activation='softmax')(x)
  model = Model(inputs=input, outputs=output)

  for layer in xception.layers[:-7]:
    layer.trainable = False
  
  model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
  
  return model

# model1 = keras.models.load_model(model_path1)
# model2 = keras.models.load_model('app/trial7_model.h5')

model = build_model()
model.load_weights(model_path1)

def preprocess_image(img):
  IMAGE_SHAPE = (299, 299,3)
  test_image = img.resize((299,299))
  test_image = image.img_to_array(test_image)
  test_image = keras.applications.xception.preprocess_input(test_image)
  test_image = np.expand_dims(test_image, axis=0)
  return test_image


def predict_bird(img):
  prediction = model.predict(img, verbose=2)
  pred = np.argmax(prediction)
  return classes[pred]
