ASL Alphabet Recognition Project
Project by Group 2 for LBYCPF3

This project implements a system that uses a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) hand gestures, specifically the ASL alphabet. The model is trained using a dataset of hand gesture images and can make predictions in real-time via webcam input. The goal of the project is to enhance accessibility and facilitate communication for individuals who use ASL.

Requirements
To run this project, you need the following Python libraries:

Python 3.2 or higher
OpenCV2
TensorFlow
NumPy
Dataset
Before running the project, you must download the dataset from the following link:

ASL Dataset: "https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data"

code : "https://github.com/ShahanMalik/ASL-detection"

After downloading, extract the dataset to a folder on your local machine.

Setup
Install dependencies:

You can install the required Python libraries using the following command:
pip install opencv-python tensorflow numpy

Prepare the dataset:
Download the dataset and extract it. Place the extracted folder in the project directory.

Running the Project
1. Train the model
To start, navigate to the project directory and run the following command to train the model:

python train_model.py
This will train the model using the ASL alphabet dataset and save the trained model as asl_cnn_model.keras.

2. Test the model
Once the model has been trained, test it using the following command:

python test_model.py
This will evaluate the model's performance on the test dataset.

3. Real-time detection
Finally, run the real-time detection script to use the trained model for ASL alphabet recognition via your webcam:

python realtime_detection.py
This script will continuously capture webcam images, process them, and display the predicted ASL letter in real time.

Output
The model predicts the ASL alphabet letter based on the input image.
In the real-time detection mode, the predicted letter will be displayed on the webcam feed.
