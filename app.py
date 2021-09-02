import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

st.title('Classification of Grain-Eating Birds')
st.header('Classifying Grain-Eating Birds')



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

def predict_bird(img):
  prediction = model1.predict(img, verbose=2)
  pred = np.argmax(prediction)
  return classes[pred]

birds = ''
for bird in classes:
    birds += f'{bird.capitalize()}, '

st.markdown(f"The birds are classified into ten different classes namely: {birds[:-2]}.")

st.write(" ")
uploaded_image = st.file_uploader(f'Upload an image of one of the above listed birds', type='jpg')
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Bird Image.', use_column_width=False)
    st.write("")
    st.write("Identifying...")
    img = preprocess_image(image)
    label = predict_bird(img)
    st.write('The image uploaded is a/an ', label.lower())




