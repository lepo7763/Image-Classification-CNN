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

# Plot some sample images
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([]) # This removes x-axis ticks
    plt.yticks([]) # This removes y-axis ticks
    plt.imshow(training_images[i], cmap=plt.cm.binary) # Display the image
    plt.xlabel(class_names[int(training_labels[i][0])])  # Display the corresponding class name
plt.show()

# Limit amount of training and testing data for efficiency
# This may affect accuracy
# Change the number based on how many images you want to train it with
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Define the convolutional neural network (CNN) model
# This CNN starts with three convolutional layers, each followed by max pooling to downsample the 
# feature maps while keeping important features. The model then flattens the 3D output to 1D and 
# passes it to a dense layer for further learning. Finally, the output layer uses softmax to 
# classify the image into one of the 10 classes.
model = models.Sequential([
    # 1st Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)), # Max Pooling layer with 2x2 pool size
    # 2nd Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), # Max Pooling layer with 2x2 pool size
    # 3rd Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(), # Flattens the 3d output to 1d for dense layers
    layers.Dense(64, activation='relu'), # Fully connected layer with 64 units
    layers.Dense(10, activation='softmax') # Output layer with 10 units for 10 classes
])

# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for 10 epochs, using validation data for testing performance during training
history = model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# This evaluates the model's performance on the testing data
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f'LOSS: {loss}')
print(f'ACCURACY: {accuracy}')

# This saves the trained model to a specified path in directory
try:
    model.save('C:/YourOwnFileDirectory') # Change to your directory
    print('Model saved successfully')
except Exception as e:
    print(f'Error saving model: {e}')

    
# This re-evaluate the model after saving to ensure consistency
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f'LOSS: {loss}')
print(f'ACCURACY: {accuracy}')
