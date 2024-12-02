import tensorflow as tf
import numpy as np
import os
import cv2

model = tf.keras.models.load_model("asl_cnn_model.keras")

class_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

test_dir = "asl_alphabet_test"
for file in os.listdir(test_dir):
    if file.endswith(".jpg"):
        img_path = os.path.join(test_dir, file)
        image = cv2.imread(img_path)
        image_resized = cv2.resize(image, (64, 64))
        image_normalized = image_resized / 255.0
        image_reshaped = np.reshape(image_normalized, (1, 64, 64, 3))

        prediction = model.predict(image_reshaped)

        predicted_index = np.argmax(prediction)
        if predicted_index < 26:
            predicted_label = class_labels[predicted_index]
            print(f"File: {file}, Predicted Label: {predicted_label}")
        else:
            print(f"File: {file}, Prediction index out of range: {predicted_index}")
