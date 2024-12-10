# **Image-Classification-CNN**

A convolutional neural network (CNN) for image classification using the CIFAR-10 dataset. This project demonstrates how to build and train a deep learning model to classify images into 10 categories, including planes, cars, birds, cats, and more.

---

## **Features**
- Built with **TensorFlow** and **Keras** for efficient deep learning implementation.
- Trained on the **CIFAR-10 dataset**.
- Achieves **70% accuracy** on the test dataset.
- Implements a three-layer CNN with max-pooling and ReLU activation.

---

## **Getting Started**

### **Dependencies**
Ensure you have the following Python libraries installed:
- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `opencv-python`
 
To install them, run:
```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

## Dataset
The CIFAR-10 dataset is automatically downloaded and loaded via Keras. Pixel values are normalized to a 0-1 range to improve training performance and model efficiency:
```python
from keras import datasets
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
```

## How to Run
Clone the repository:
```bash
git clone https://github.com/lepo7763/Image-Classification-CNN.git
cd Image-Classification-CNN
```
- Note: Remember to change folder directories within both scripts

## Training the model:
```bash
python trainCNNModel.py
```
This script will:
- `Normalize and preprocess the data`
- `Train the CNN for 10 epochs (default)`
- `Output the loss and accuracy metrics`
- `Save the trained model to your specified directory`

## Running the saved model:
```bash
python runSavedModel.py
```
- Ensure you have a saved ```.keras``` file
- Update the image file path in runSavedModel.py to test custom predictions
