import ee
import numpy as np
import streamlit as st
from PIL import Image
from skimage.transform import resize
from tensorflow import keras
import requests

# Initialize the Earth Engine Python API
ee.Initialize()

# Load the saved CNN model
cnn_model = keras.models.load_model("cnn_model.h5")

# Define the list of land use classes
land_use_classes = ["annual_crop", "forest", "herbaceous_vegetation", "highway",
                    "industrial", "pasture", "permanent_crop", "residential",
                    "river", "sea_lake"]

# Define the Earth Engine image collection to retrieve satellite imagery
image_collection = ee.ImageCollection("COPERNICUS/S2")


# Function to retrieve an image from Earth Engine based on coordinates
def get_image_from_ee(latitude, longitude):
    point = ee.Geometry.Point(longitude, latitude)
    image = image_collection.filterBounds(point).first()
    return image


# Function to preprocess the Earth Engine image for classification
def preprocess_image(image):
    image_url = image.getThumbURL({'dimensions': '64x64'})
    image_np = np.array(Image.open(requests.get(image_url, stream=True).raw))
    image_resized = resize(image_np, (64, 64), mode='reflect', anti_aliasing=True)
    return image_resized


# Function to perform land use and land cover classification using the CNN model
def classify_image(image):
    image = np.expand_dims(image, axis=0)
    predicted_class = np.argmax(cnn_model.predict(image))
    return land_use_classes[predicted_class]


# Streamlit web app
def main():
    st.title("Land Use and Land Cover Classification")

    # Get user input for coordinates
    latitude = st.number_input("Enter Latitude")
    longitude = st.number_input("Enter Longitude")

    # Retrieve the image from Earth Engine
    image = get_image_from_ee(latitude, longitude)

    if image is not None:
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Classify the image
        predicted_class = classify_image(processed_image)

        # Display the image and predicted class
        st.image(processed_image.astype('uint8'), caption="Input Image", use_column_width=True)
        st.write("Predicted Land Use Class:", predicted_class)
    else:
        st.write("No image found for the provided coordinates.")


if __name__ == '__main__':
    main()
