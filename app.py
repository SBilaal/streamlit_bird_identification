import streamlit as st
from PIL import Image, ImageOps
from model import classes, preprocess_image, predict_bird


st.title('Classification of Grain-Eating Birds')
st.header('Classifying Grain-Eating Birds')


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



