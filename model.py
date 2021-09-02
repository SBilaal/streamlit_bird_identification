from tensorflow.keras.preprocessing import image
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image, ImageOps
import numpy as np
import os

# from google.colab import drive
# drive.mount('/content/gdrive')

# data = '/content/gdrive/MyDrive/Birds_Test_Dataset'
test_path = 'test'
classes = ['AFRICAN FIREFINCH', 'CROWNED PIGEON', 'GREEN JAY', 'MOURNING DOVE', 'NICOBAR PIGEON', 'PURPLE FINCH', 'RED BROWED FINCH', 'ROCK DOVE', 'STRAWBERRY FINCH', 'YELLOW HEADED BLACKBIRD']
# print(test_path)

os.path.isdir(test_path)
root = 'D:\\MTE 500\Project\\Birds_Test_Dataset'

model1 = keras.models.load_model(f'{root}\\models\\trial7_model.h5' )

def preprocess_image(img):
  IMAGE_SHAPE = (299, 299,3)
  test_image = img.resize((299,299))
  test_image = image.img_to_array(test_image)
  test_image = keras.applications.xception.preprocess_input(test_image)
  test_image = np.expand_dims(test_image, axis=0)
  return test_image

# img = preprocess_image(f'{root}\\{test_path}\\{classes[7]}\\4.jpg')

def predict_bird(img):
  prediction = model1.predict(img, verbose=2)
  pred = np.argmax(prediction)
  return classes[pred]
