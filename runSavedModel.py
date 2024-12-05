import tensorflow 
from tensorflow import keras
from keras import datasets, layers, models
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 training images and labels from datasets within the Keras library
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
# This helps the neural network train faster and perform better
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Define class names for CIFAR-10 dataset
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load the pre-trained model from the specified file path
# This should be a .keras file
model = models.load_model(r"C:YourOwnFileDirectory") # Change to your directory

# Load and preprocess an image for prediction
# The image is read from the specified path 
img = cv.imread(r"C:YourOwnFileDirectory") # Change to your directory

# This checks if the image was successfully loaded
if img is None:
    raise FileExistsError(f'Failed to load image file: {img}')

# This converts the image from BGR to RGB format (OpenCV loads images in BGR by default)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# This resizes the image 
img = cv.resize(img, (32, 32))

# This displays the image using matplotlib to verify the loaded image
plt.imshow(img, cmap=plt.cm.binary)
plt.show()


# Make a prediction using the loaded model
# The image is normalized by dividing by 255 to match the training scale
prediction = model.predict(np.array([img]) / 255)

# Get the index of the class with the highest probability
index = np.argmax(prediction)

print(f'Prediction is {class_names[index]}')

