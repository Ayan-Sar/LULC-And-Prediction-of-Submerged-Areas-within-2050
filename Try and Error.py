import numpy as np
import os
import random
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
import joblib

# Define the path to the dataset directory
dataset_path = "D:\\Research Paper Codes\\LULC and Prediction\\EuroSAT Dataset\\EuroSAT"

# Define the list of land use classes
land_use_classes = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                    "Industrial", "Pasture", "PermanentCrop", "Residential",
                    "River", "SeaLake"]

# Load and preprocess the dataset
def load_dataset():
    images = []
    labels = []
    for land_use in land_use_classes:
        class_path = os.path.join(dataset_path, land_use)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = imread(image_path)
            image = resize(image, (64, 64), mode='reflect', anti_aliasing=True)
            images.append(image)
            labels.append(land_use_classes.index(land_use))
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load the dataset
images, labels = load_dataset()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(land_use_classes), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
_, cnn_accuracy = model.evaluate(X_test, y_test)
print("CNN Accuracy:", cnn_accuracy)

# Save the CNN model
model.save("cnn_model.h5")

# Reshape the image data for RF model
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

# Create and train the RF model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train)

# Predict the labels for test set
y_pred_rf = rf_model.predict(X_test_rf)

# Evaluate the RF model
rf_accuracy = np.mean(y_pred_rf == y_test)
print("RF Accuracy:", rf_accuracy)

# Save the RF model
joblib.dump(rf_model, "rf_model.pkl")

# Print classification report for RF model
print(classification_report(y_test, y_pred_rf, target_names=land_use_classes))