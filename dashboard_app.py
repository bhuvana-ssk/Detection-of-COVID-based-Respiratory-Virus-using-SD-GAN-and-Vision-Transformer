import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('F:\Dashboard\TrainedModel.h5')

class_labels = ['COVID19', 'PNEUMONIA', 'NORMAL']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

st.title('Detection of COVID-19 on CXR Images')
st.write('Upload an X-ray Image here')

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded X-ray Image.', width=200)

    if st.button('Classify'):
        img_array = preprocess_image(uploaded_file)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]

        st.success(f"Prediction: {predicted_label}")

if __name__ == '__main__':
    st.write("")
