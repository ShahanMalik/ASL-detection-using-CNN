import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("asl_cnn_model.keras")

class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    resized = resized.astype('float32') / 255.0
    
    rgb_image = np.stack([resized] * 3, axis=-1)
    
    input_image = np.expand_dims(rgb_image, axis=0)

    prediction = model.predict(input_image)

    predicted_index = np.argmax(prediction)

    if predicted_index < len(class_labels):
        predicted_label = class_labels[predicted_index]
    else:
        predicted_label = "Unknown"

    cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Real-time ASL Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
