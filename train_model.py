import os # reading and writing to a file 
from glob import glob # original code
# Used to retrieve files and directories that match a specified pattern. It helps in finding all the image files in a directory
import numpy as np # used for numerical operations and handling image and label data.
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast, 
    RandomGamma, Resize, Normalize, RandomCrop, HueSaturationValue, CLAHE, GaussianBlur,
    RandomShadow, RandomRain, RandomFog
) # used for data augmentation 
import cv2 # read, write, and process images, including converting color spaces 
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB5 # It is used as a pre-trained model for feature extraction.
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard #addition of ReduceLROnPlateau from original code
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization #addition of BatchNormalization from original code
from tensorflow.keras.models import Sequential # original code -> simple way to build a neural network.
from tensorflow.keras.optimizers import AdamW # implements Adam algorithm with weight decay. helps in optimizing the model's parameters
from sklearn.model_selection import train_test_split #It is helping in dividing the dataset into training and validation sets.
from sklearn.metrics import classification_report # Builds a text report showing the main classification metrics
from keras_tuner import HyperModel # A base class for defining hypermodels for hyperparameter tuning.
from keras_tuner.tuners import RandomSearch #It helps in finding the best hyperparameters for the model.
from tqdm import tqdm # original code -> used to display progress bars during feature extraction and other iterative processes.
from utils import load_dataset # utility function for loading the dataset, involves reading images and labels from disk and preprocessing them 

# This function performs a series of transformations that will be applied to the training images to augment the data. 
# This to produce more diverse data for training the model. 
# Extra Implementation - Data Augmentation
# Define the augmentation pipeline
def get_train_transforms():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=90, p=0.5),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.5),
        RandomCrop(256, 256, p=0.5),
        HueSaturationValue(p=0.5),
        CLAHE(p=0.5),
        GaussianBlur(p=0.5),
        RandomShadow(p=0.5),
        RandomRain(p=0.5),
        RandomFog(p=0.5),
        Resize(299, 299),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# Convert the image to a tensor with the correct shape using NumPy
def to_tensor(image):
    return image.astype(np.float32)

def augment_image(image, transform):
    augmented = transform(image=image)
    image = augmented['image']
    return image

# Custom data generator for augmented images
class AugmentedDataGenerator(tf.keras.utils.Sequence): # Initializes the generator with image paths, labels, batch size, transformation pipeline, and shuffle flag
    def __init__(self, image_paths, labels, batch_size, transform, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle # Shuffles the data at the end of each epoch if shuffle is True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in batch_indices]
        batch_labels = [self.labels[k] for k in batch_indices]

        images = []
        for img_path in batch_image_paths:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented_image = augment_image(image, self.transform)
            tensor_image = to_tensor(augmented_image)
            images.append(tensor_image)

        X = np.stack(images)
        y = np.stack(batch_labels)

        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

# Load the data for training. From original code
train_files, train_targets = load_dataset("C:/Users/naash/Documents/programming/skin-cancer-classification-main/train")
valid_files, valid_targets = load_dataset("C:/Users/naash/Documents/programming/skin-cancer-classification-main/valid")

# Ensure the number of train files matches the number of train targets -> to prevent data inconsistency.
if len(train_files) != len(train_targets):
    raise ValueError("Mismatch between number of training files and training targets")

 
# Extracts label names from the directory structure. The item[11:-1]: assumes the label name is found between certain indices in the directory string.
# Load labels. From original code
label_name = [item[11:-1] for item in sorted(glob("C:/Users/naash/Documents/programming/skin-cancer-classification-main/train/*/"))]

# Summary of the dataset. From original code
print("Train Files Size: {}".format(len(train_files)))
print("Train Files Shape: {}".format(train_files.shape))
print("Target Shape: {}".format(train_targets.shape))
print("Label Names: {}".format(label_name))

#Extra implementation 
# Initialize the data generators
# Calling the method get_train_transforms and passing to the class 
# AugmentedDataGenerator. This Initializes the data generators for training and validation 
# data with the specified batch size and transformation pipeline.
train_transform = get_train_transforms()
train_generator = AugmentedDataGenerator(train_files, train_targets, batch_size=32, transform=train_transform)
valid_transform = get_train_transforms()
valid_generator = AugmentedDataGenerator(valid_files, valid_targets, batch_size=32, transform=valid_transform, shuffle=False)

#Extra implementation
# Define the feature extraction model
# Loading EfficientNetB5
transfer_model = EfficientNetB5(include_top=False, weights="imagenet", input_shape=(299, 299, 3))

#Extra implementation
# Gradual unfreezing of layers
for layer in transfer_model.layers:
    layer.trainable = False
for layer in transfer_model.layers[-100:]:  # Unfreeze the last 100 layers for more fine-tuning
    layer.trainable = True

# Creates a feature extraction model from the EfficientNetB5 base model.
# inputs: Input tensor of the EfficientNetB5 model.
# outputs: Output tensor of the EfficientNetB5 model without the top layer.
feature_extractor = tf.keras.Model(inputs=transfer_model.input, outputs=transfer_model.output)

# Extra implementation - Hyperparameter tuning -> class defines a model structure that will be used to search for the best hyperparameters using Keras Tuner.
# Define the HyperModel for hyperparameter tuning
class MyHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=(10, 10, 2048)))  # EfficientNetB5 output shape #original code
        
        model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))
        model.add(BatchNormalization())
        model.add(Dense(hp.Int('units_1', 512, 2048, step=512), activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
        model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))
        model.add(BatchNormalization())
        model.add(Dense(hp.Int('units_2', 512, 1024, step=256), activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
        model.add(Dropout(hp.Float('dropout_3', 0.2, 0.5, step=0.1)))
        model.add(BatchNormalization())
        model.add(Dense(hp.Int('units_3', 128, 512, step=128), activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
        model.add(Dropout(hp.Float('dropout_4', 0.2, 0.5, step=0.1)))
        model.add(BatchNormalization())
        model.add(Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01)))

        model.compile(
            loss="categorical_crossentropy",
            optimizer=AdamW(
                learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='LOG')),
            metrics=["accuracy"])
        
        return model
    
# The RandomSearch tuner will be used to find the best hyperparameters by randomly sampling from the search space defined in MyHyperModel.
# Hyperparameter tuning
hypermodel = MyHyperModel()

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=30, 
    executions_per_trial=1,
    directory='new_dir',  
    project_name='hyperparameter_tuning')

tuner.search_space_summary() #Prints the summary of the search space for hyperparameter tuning.

# Extract features for hyperparameter tuning
def extract_features_for_tuning(generator, feature_extractor):
    features = []
    labels = []

    for batch_data, batch_labels in tqdm(generator, desc="Extracting features"):
        batch_features = feature_extractor.predict(batch_data)
        features.append(batch_features)
        labels.append(batch_labels)

    features = np.vstack(features)
    labels = np.vstack(labels)

    return features, labels

print("[INFO] Extracting features for hyperparameter tuning using EfficientNetB5 model...")
train_data_tuning, train_labels_tuning = extract_features_for_tuning(train_generator, feature_extractor) # Extracted features and labels for training data
valid_data_tuning, valid_labels_tuning = extract_features_for_tuning(valid_generator, feature_extractor) # Extracted features and labels for validation data.

#Extra implementation 
# Perform the search
tuner.search(train_data_tuning, train_labels_tuning, epochs=20, validation_data=(valid_data_tuning, valid_labels_tuning))

#Extra implementation
# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"The best hyperparameters are: {best_hps.values}")

#Extra implementation
# Define callbacks
checkpointer = ModelCheckpoint(
    filepath="saved_models_weights_checkpointer/weights.best.model.hdf5", 
    verbose=1, 
    save_best_only=True
)
# Initializes callbacks for model checkpointing, early stopping, learning rate reduction, and TensorBoard logging.
early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min", restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
tensorboard = TensorBoard(log_dir="logs")

# Compute class weights to handle class imbalance
class_weights = {i: weight for i, weight in enumerate(np.max(train_targets.sum(axis=0)) / train_targets.sum(axis=0))}

# Function to create and compile the model using the best hyperparameters. Improved from the original source code. 
def create_model_with_best_hps(best_hps):
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=(10, 10, 2048)))  # EfficientNetB5 output shape
    model.add(Dropout(best_hps.get('dropout_1')))
    model.add(BatchNormalization())
    model.add(Dense(best_hps.get('units_1'), activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
    model.add(Dropout(best_hps.get('dropout_2')))
    model.add(BatchNormalization())
    model.add(Dense(best_hps.get('units_2'), activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
    model.add(Dropout(best_hps.get('dropout_3')))
    model.add(BatchNormalization())
    model.add(Dense(best_hps.get('units_3'), activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
    model.add(Dropout(best_hps.get('dropout_4')))
    model.add(BatchNormalization())
    model.add(Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=AdamW(learning_rate=best_hps.get('learning_rate'), clipnorm=1.0),
        metrics=["accuracy"])

    return model

# Train the model
# Splits the training data for validation and trains the model using the best hyperparameters with the specified callbacks and class weights.
print("\n[INFO] Training the model...\n")
X_train, X_val, y_train, y_val = train_test_split(train_data_tuning, train_labels_tuning, test_size=0.2, random_state=42, stratify=np.argmax(train_labels_tuning, axis=1))

model = create_model_with_best_hps(best_hps) #improvised from the original source code. 
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[checkpointer, early_stopping, reduce_lr, tensorboard],
    class_weight=class_weights,
    verbose=1
)

#Extra implementation 
# Evaluate the model -> on the validation data and prints the classification report.
val_preds = model.predict(valid_data_tuning)
val_preds = np.argmax(val_preds, axis=1)
val_labels = np.argmax(valid_labels_tuning, axis=1)
print(classification_report(val_labels, val_preds, target_names=label_name))

""" Save the model """ # from original source code.
# Save the final model weights and configuration
model.save_weights("C:/Users/naash/Documents/programming/skin-cancer-classification-main/weights/CNN_model.h5")
model_json = model.to_json()
with open("C:/Users/naash/Documents/programming/skin-cancer-classification-main/models/CNN_model.json", "w") as json_file:
    json_file.write(model_json)

print("\nSaved model weights and configuration to disk.\n")




 

 



  



