import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import tensorflow as tf
from tensorflow.keras.models import load_model  # Use the Keras integrated in TensorFlow
import cv2  # Install opencv-python
import numpy as np


# We add these imports from the snippet:
from PIL import Image, ImageOps  # Install pillow


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


from tensorflow.keras.layers import DepthwiseConv2D


# Custom DepthwiseConv2D to ignore 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
   def __init__(self, *args, **kwargs):
       kwargs.pop("groups", None)
       super().__init__(*args, **kwargs)


# Load the model 
model = load_model(
   "/home/cecegraf/vsCode/Python/Classwork/DogAI/keras_model.h5",
   compile=False,
   custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
)


# Load the labels 
class_names = open("/home/cecegraf/vsCode/Python/Classwork/DogAI/labels.txt", "r").readlines()


# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)


previous_key = None  # Track the last key pressed


while True:
   # Grab the webcamera's image
   ret, image = camera.read()


   # Resize the raw image into (224-height,224-width) pixels
   image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


   # Show the image in a window
   cv2.imshow("Webcam Image", image)


   # Make the image a numpy array and reshape it to the model's input shape.
   image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
   # Normalize the image array
   image = (image / 127.5) - 1


   # Predict using the webcam frame
   prediction = model.predict(image)
   index = np.argmax(prediction)
   class_name = class_names[index]
   confidence_score = prediction[0][index]


   # Print prediction and confidence score
   print("Class:", class_name[2:], end="")
   print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")


   # Listen to the keyboard for presses.
   keyboard_input = cv2.waitKey(1)


   # Quit when pressing '1' then '3' in sequence
   if keyboard_input == ord('1'):
       previous_key = '1'
   elif keyboard_input == ord('3') and previous_key == '1':
       print("User pressed 1 then 3 — exiting!")
       break
   elif keyboard_input == 27:  # ESC key
       break
   elif keyboard_input != -1:
       # If a different key is pressed, reset
       previous_key = None


   if class_name == "0 Dogs\n":
       message_to_send = "ON\n"
   elif class_name =="1 Not Dogs\n":
    message_to_send = "OFF\n"




camera.release()
cv2.destroyAllWindows()


# -------------------------
# Below is the snippet logic for classifying a single image:



def classify_single_image(image_path):
   # Create the array of the right shape for our model
   data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


   # Open the image with Pillow
   image = Image.open(image_path).convert("RGB")


   # Resize the image to 224x224
   size = (224, 224)
   image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)


   # Convert the image to a numpy array
   image_array = np.asarray(image)


   # Normalize the image
   normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1


   # Load the image into the array
   data[0] = normalized_image_array


   # Predict using the same loaded model
   prediction = model.predict(data)
   index = np.argmax(prediction)
   class_name = class_names[index]
   confidence_score = prediction[0][index]


   # Print prediction and confidence score
   print("Class:", class_name[2:], end="")
   print("Confidence Score:", confidence_score)


 







