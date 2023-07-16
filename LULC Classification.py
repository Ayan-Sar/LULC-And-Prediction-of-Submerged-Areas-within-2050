import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
# Step 1: Data Preprocessing
dataset_dir = 'D:\\Research Paper Codes\\LULC and Prediction\\EuroSAT Dataset\\EuroSAT'

# Define the list of land use classes
land_use_classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture',
                    'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Initialize lists to store images and labels
images = []
labels = []

# Loop over the land use classes
for class_index, class_name in enumerate(land_use_classes):
    # Get the path to the class directory
    class_dir = os.path.join(dataset_dir, class_name)

    # Loop over the images in the class directory
    for image_name in os.listdir(class_dir):
        # Get the path to the image file
        image_path = os.path.join(class_dir, image_name)

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Resize the image to a fixed size (e.g., 64x64 pixels)
        image = cv2.resize(image, (64, 64))

        # Append the image and corresponding label to the lists
        images.append(image)
        labels.append(class_index)

# Convert the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Print the shape of the loaded dataset
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize the image data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels into numerical format using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert numerical labels to one-hot encoding
num_classes = len(label_encoder.classes_)
y_train_onehot = to_categorical(y_train_encoded, num_classes)
y_test_onehot = to_categorical(y_test_encoded, num_classes)

# Step 2: CNN Model

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 13)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_onehot, batch_size=32, epochs=10, validation_data=(X_test, y_test_onehot))

# Step 3: Random Forest (RF) Model

# Reshape the data for RF model (assuming images are flattened)
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

# Create the RF model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the RF model
rf_model.fit(X_train_rf, y_train_encoded)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test_rf)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test_encoded, y_pred_rf)

# Print RF model evaluation metrics
print("Random Forest Model Evaluation:")
print(classification_report(y_test_encoded, y_pred_rf))
print("Accuracy:", accuracy_rf)