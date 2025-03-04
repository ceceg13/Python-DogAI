import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import serial
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps

# Custom DepthwiseConv2D to ignore the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

# Load the model and labels (adjust paths as needed)
model = load_model(
    "/home/cecegraf/vsCode/Python/Classwork/DogAI/keras_model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
)
class_names = open("/home/cecegraf/vsCode/Python/Classwork/DogAI/labels.txt", "r").readlines()

# Initialize camera
print("Attempting to open the camera...")
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Camera not opened!")
else:
    print("Camera opened successfully!")

# Initialize Arduino serial connection
print("Attempting to connect to Arduino...")
arduino = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)  # Allow time for Arduino to reset
print("Arduino connected successfully!")





while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to grab frame from camera")
        break
    cv2.imshow("Webcam Image", frame)

    # Preprocess the frame for the model
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Make prediction
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Debug prints to see what the model is outputting
    print("Raw label:", repr(class_name))
    predicted_label = class_name
    print("Predicted label:", predicted_label)
    print("Confidence Score:", confidence_score)

    # Send "ON" if the label exactly equals "0 Dogs"; otherwise, "OFF"
    if predicted_label == "0 Dogs\n":
        message_to_send = "ON\n"
  

    # Send command only if it differs from the last sent command
   
    arduino.write(message_to_send.encode('utf-8'))
    print(f"Sent to Arduino: {message_to_send.strip()}")
    time.sleep(0.2)
    # Optionally, check for response from Arduino
    if arduino.in_waiting > 0:
        received_data = arduino.readline().decode('utf-8').strip()
        print(f"Received from Arduino: {received_data}")

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # ESC key
        print("ESC pressed. Exiting loop.")
        break

    

arduino.close()
print("Serial port closed")
camera.release()
cv2.destroyAllWindows()
