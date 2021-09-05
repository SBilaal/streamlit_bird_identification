import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from model import classes, preprocess_image, predict_bird


st.title('Identification of Grain-Eating Birds')
st.header('Identifying Grain-Eating Birds')


birds = ''
for bird in classes:
    birds += f'{bird.capitalize()}, '

st.markdown(f"The grain-eating birds picked to be identified by the model are: {birds[:-2]}.")

st.write(" ")
uploaded_image = st.file_uploader(f'Upload an image of one of the above listed birds', type=['png', 'jpg', 'jfif', 'jpeg'])
if uploaded_image is not None:
    image = Image.open(uploaded_image)

    #Checks if channel is 4 and converts to 3.
    image_array = np.array(image)
    channel = image_array.shape[-1]
    if channel == 4:
        img = img.convert('RGBA')
        background = Image.new('RGBA', img.size, (255,255,255))
        image = Image.alpha_composite(background, img).convert('RGB')

    st.image(image, caption='Uploaded Bird Image.', use_column_width=False)
    st.write("")
    st.write("Identifying...")
    img = preprocess_image(image)
    label = predict_bird(img)
    st.write('The image uploaded is a/an ', label.lower())



