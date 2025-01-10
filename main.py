from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
import os

# Print the number of GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Force TensorFlow to use CPU (uncomment the next line if you don't want to use GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the pre-trained model
try:
    model = load_model("face_recog.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the class labels
try:
    with open("labels.txt", "r") as file:
        class_names = file.readlines()
except FileNotFoundError:
    print("Error: 'labels.txt' not found. Make sure the file is in the same directory.")
    exit()

# Initialize the webcam
camera = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Could not open the camera. Please check if it's connected or accessible.")
    exit()

print("Press 'Esc' to exit the program.")

while True:
    # Capture an image from the webcam
    ret, image = camera.read()

    if not ret:
        print("Error: Failed to capture image from the camera.")
        break

    # Resize the image to the model's input size
    try:
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
        break

    # Show the live webcam feed
    cv2.imshow("Webcam Image", image)

    # Prepare the image for prediction
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1  # Normalize the image array

    # Make a prediction
    try:
        prediction = model.predict(image_array, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
    except Exception as e:
        print(f"Error during prediction: {e}")
        break

    # Print the prediction and confidence score
    print(f"Class: {class_name}, Confidence Score: {np.round(confidence_score * 100, 2)}%")

    # Exit the program when the 'Esc' key is pressed
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # ASCII for 'Esc'
        print("Exiting program.")
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
