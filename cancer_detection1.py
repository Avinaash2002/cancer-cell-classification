import os
import cv2
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from glob import glob

# Load the model architecture
def load_model():
    print("[INFO] loading CNN Model...")
    base_model = InceptionV3(include_top=False, weights="imagenet", input_tensor=Input(shape=(299, 299, 3)))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(3, activation="softmax")(x)
    
    # Define the complete model
    model = Model(inputs=base_model.input, outputs=x)
    
    # Load weights into the model
    try:
        model.load_weights("C:/Users/naash/Documents/programming/skin-cancer-classification-main/weights/CNN_model.h5")
        print("[INFO] Loaded model from disk.")
    except ValueError as e:
        print("[ERROR] Error loading weights: ", e)
        print("[INFO] Ensure the architecture matches the saved model's architecture.")
    return model

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Get label names
def get_label_names():
    train_dir = "C:/Users/naash/Documents/programming/skin-cancer-classification-main/train"
    if not os.path.exists(train_dir) or not os.path.isdir(train_dir):
        raise ValueError(f"[ERROR] Training directory '{train_dir}' does not exist or is not a directory.")
    label_names = [os.path.basename(os.path.normpath(d)) for d in sorted(glob(train_dir + "/*/"))]
    return label_names

# Predict the image
def predict_image(model, image_path):
    tensor = load_and_preprocess_image(image_path)
    print(f"[DEBUG] Input tensor shape: {tensor.shape}")  # Debugging statement
    prediction = model.predict(tensor)
    print(f"[DEBUG] Prediction: {prediction}")  # Debugging statement
    return prediction

# Annotate the image
def annotate_image(image_path, label, prob):
    orig = cv2.imread(image_path)
    cv2.putText(
        orig, "Label: {}, {:.2f}%".format(label, prob * 100), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2
    )
    result_path = os.path.join('static', 'cancer image', 'result.jpg')
    cv2.imwrite(result_path, orig)
    if os.path.exists(result_path):
        print(f"[INFO] Result image saved successfully at {result_path}")
    else:
        print(f"[ERROR] Failed to save result image at {result_path}")
    return result_path











