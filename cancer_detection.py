import argparse
from glob import glob
import os
import cv2
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())

# Input image
input_image = args["image"]

# Transfer learning using Inception V3
# Load the Inception V3 model as well as the network weights from disk
print("[INFO] loading {}...".format("CNN Model"))
base_model = InceptionV3(include_top=False, weights="imagenet", input_tensor=Input(shape=(299, 299, 3)))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dense(3, activation="softmax")(x)  # Removed Dropout layers

# Define the complete model
CNN_model = Model(inputs=base_model.input, outputs=x)

# Load weights into the model
try:
    CNN_model.load_weights("weights/CNN_model.h5")
    print("[INFO] Loaded model from the disk.")
except ValueError as e:
    print("[ERROR] Error loading weights: ", e)
    print("[INFO] Ensure the architecture matches the saved model's architecture.")

# Prediction
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

tensor = load_and_preprocess_image(input_image)
prediction = CNN_model.predict(tensor)

# Debug: Print the prediction
print("[DEBUG] Prediction array:", prediction)

# Verify directory structure and load labels correctly
train_dir = "C:/Users/naash/Documents/programming/skin-cancer-classification-main/train"
if not os.path.exists(train_dir) or not os.path.isdir(train_dir):
    raise ValueError(f"[ERROR] Training directory '{train_dir}' does not exist or is not a directory.")

# Extract label names from the directory structure
label_names = [os.path.basename(os.path.normpath(d)) for d in sorted(glob(train_dir + "/*/"))]

print("[INFO] Analyzing the skin lesion.")
print("[INFO] Please Wait...")

# Show output
cv2.namedWindow("Classification", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Classification", 1920, 1080)
orig = cv2.imread(input_image)

# Check if the prediction is within the bounds of label_names
label_index = np.argmax(prediction)
if label_index < len(label_names):
    label = label_names[label_index]
    prob = prediction[0][label_index]
    print("[INFO] Analysis Completed!")
    print("[INFO] {} detected in the image.".format(label))
    cv2.putText(
        orig, "Label: {}, {:.2f}%".format(label, prob * 100), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2
    )
    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
else:
    print("[ERROR] Label index out of range. Check the label names and prediction output.")


