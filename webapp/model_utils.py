"""
Model utilities for Arthritis X-ray Classification
This module contains functions to build and load the trained model
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io


# Class labels (11 classes after dropping 'soft tissue calcination')
CLASS_LABELS = [
    "Class 0",
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4",
    "Class 5",
    "Class 6",
    "Class 7",
    "Class 8",
    "Class 9",
    "Class 10"
]


def build_model(input_shape=(128, 128, 1), num_labels=11):
    """
    Build the CNN model architecture matching the trained model

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_labels: Number of output labels (11 for this model)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
               kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),

        Dense(num_labels, activation='sigmoid')  # Multi-label classification
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def preprocess_image(image_file, target_size=(128, 128)):
    """
    Preprocess uploaded image for model prediction

    Args:
        image_file: File object from Flask request
        target_size: Target size for resizing (width, height)

    Returns:
        Preprocessed numpy array ready for prediction
    """
    # Read image file
    image = Image.open(io.BytesIO(image_file.read()))

    # Convert to grayscale
    image = image.convert('L')

    # Resize to target size
    image = image.resize(target_size)

    # Convert to numpy array
    img_array = img_to_array(image)

    # Normalize pixel values (0-1)
    img_array = img_array / 255.0

    # Reshape to add batch dimension
    img_array = img_array.reshape(1, target_size[0], target_size[1], 1)

    return img_array


def predict_arthritis(model, processed_image, threshold=0.9):
    """
    Make prediction on preprocessed image

    Args:
        model: Trained Keras model
        processed_image: Preprocessed image array
        threshold: Probability threshold for binary classification

    Returns:
        Dictionary with predictions and probabilities
    """
    # Get model predictions
    predictions = model.predict(processed_image, verbose=0)

    # Get probabilities for each class
    probabilities = predictions[0]

    # Apply threshold to get binary predictions
    binary_predictions = (probabilities > threshold).astype(int)

    # Create results dictionary
    results = {
        'detected_conditions': [],
        'all_predictions': []
    }

    for idx, (prob, binary) in enumerate(zip(probabilities, binary_predictions)):
        class_result = {
            'class': CLASS_LABELS[idx],
            'class_id': idx,
            'probability': float(prob),
            'detected': bool(binary)
        }
        results['all_predictions'].append(class_result)

        if binary:
            results['detected_conditions'].append(class_result)

    # Sort detected conditions by probability
    results['detected_conditions'].sort(key=lambda x: x['probability'], reverse=True)

    return results
