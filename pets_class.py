import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os

#--------------------------------------------------------------------------------
# UPLOAD image
st.title("🐈CATS OR DOGS🐶")
st.header("""CLASSIFICATION USINGS CNN """)
#--------------------------------------------------------------------------------
# Create an image upload widget
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Check if the "Get Prediction" button is clicked
    if st.button("Get Prediction"):
        # Load and preprocess the image
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (300, 300))
        image = np.expand_dims(image, axis=0)
#--------------------------------------------------------------------------------
        # Load your trained model
        model = tf.keras.models.load_model('dogs_cats_model1')
#--------------------------------------------------------------------------------
        # Make a prediction
        predictions = model.predict(image)

        threshold = 0.5

        if predictions[0][0] >= threshold:
            predicted_class = 'dogs'  
        else:
            predicted_class = 'cats'  

        st.write(f"Predicted Class: {predicted_class}")  

        if predicted_class == 'cats':
            st.image('https://media.tenor.com/LCRUpy3tJpEAAAAC/cat-i-am.gif')
        else:
            st.image('https://y.yarn.co/67d8263c-c566-4ddc-947c-4ad15847172a_text.gif')