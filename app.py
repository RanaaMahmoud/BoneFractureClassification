import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st

# Load the saved model
model_path = 'E:/MediVision-AI/Bone Fracture Classification_streamlit/bonefracture_classification_model (1).h5'
model = tf.keras.models.load_model(model_path)

def load_and_preprocess_image(img, target_size=(180, 180)):
    # Load the image with the target size
    img = img.resize(target_size)
    
    # Convert the image to an array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the model's input shape (1, IMG_HEIGHT, IMG_WIDTH, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image (assuming your model was trained with rescaling of 1./255)
    img_array /= 255.0
    
    return img_array

class_names = {
    0 : 'Fractured',
    1 : 'Not fractured',
}

def make_prediction(model, img, class_names):
    # Preprocess the image
    img_array = load_and_preprocess_image(img)
    
    # Perform the prediction
    prediction = model.predict(img_array)
    
    # Since it's a binary classification, we use a threshold of 0.5
    class_index = 1 if prediction[0] > 0.5 else 0
    class_label = class_names[class_index]
    
    return class_label, prediction[0]

# Streamlit interface
st.title("Bone Fracture Detection")
st.write("Upload an X-ray image to detect if there is a fracture.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    st.write("Classifying...")
    class_label, confidence = make_prediction(model, img, class_names)
    
    # Display prediction and confidence
    st.write(f"**Predicted Class**: {class_label}")
    st.write(f"**Confidence**: {confidence[0]:.2f}")

# To run the app, save the file and use the following command in the terminal:
# streamlit run app.py
