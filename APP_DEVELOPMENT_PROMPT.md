# Arthritis Diagnosis App - Development Prompt

## Project Overview
Build a web application that uses a trained CNN model to diagnose arthritis conditions from X-ray images. The app should accept X-ray images as input and output which arthritis conditions are detected.

## Technical Context

### Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input**: Grayscale X-ray images (128×128 pixels)
- **Output**: Multi-label classification for 11 arthritis conditions
- **Accuracy**: 93.75% on test data
- **Model File**: Located at `Codes/Image_analysis_multiclass_balanced.ipynb`

### Model Specifications
- **Input preprocessing**:
  - Convert image to grayscale
  - Resize to 128×128 pixels
  - Normalize pixel values (divide by 255 to get 0-1 range)
  - Reshape to (128, 128, 1) with channel dimension

- **Output processing**:
  - Model outputs 11 probability scores (0-1 range)
  - Apply threshold of 0.9 to convert to binary predictions
  - Each output position represents a different arthritis condition
  - Multiple conditions can be detected simultaneously (multi-label)

### Conditions Detected
The model detects 11 different arthritis-related conditions. The original dataset includes:
- 10 arthritis manifestations
- Note: "Soft tissue calcination" was excluded from the balanced model due to class imbalance

**Dataset Source**: [Roboflow X-ray Rheumatology Dataset](https://universe.roboflow.com/roboflow-100/x-ray-rheumatology)

## Application Requirements

### Core Functionality
1. **Image Upload**
   - Accept X-ray image uploads (JPG, PNG formats)
   - Support drag-and-drop or file selection
   - Display uploaded image preview

2. **Image Processing**
   - Convert uploaded image to grayscale
   - Resize to 128×128 pixels
   - Normalize pixel values (0-1 range)
   - Add channel dimension for model input

3. **Model Inference**
   - Load the trained CNN model
   - Pass preprocessed image through the model
   - Get probability predictions for all 11 conditions

4. **Results Display**
   - Show detected conditions (those with probability > 0.9)
   - Display confidence scores for each detected condition
   - Present results in a user-friendly format (list, cards, or visual indicators)
   - Option to show all 11 conditions with their probabilities

### User Interface
- **Clean and intuitive design**
- **Upload section**: Clear call-to-action for image upload
- **Preview section**: Show the uploaded X-ray image
- **Results section**: Display diagnosis results prominently
- **Optional features**:
  - Side-by-side comparison (original image + results)
  - Download results as PDF/report
  - History of previous analyses (if user authentication added)

### Technology Stack Options

**Option 1: Web App (Recommended)**
- **Frontend**: React, Vue.js, or plain HTML/CSS/JavaScript
- **Backend**: Flask (Python) or FastAPI
- **Model Serving**: TensorFlow.js (browser) or Python backend
- **Deployment**: Heroku, Vercel, or AWS

**Option 2: Desktop App**
- **Framework**: Tkinter (Python), Electron, or PyQt
- **Model**: Load directly with TensorFlow/Keras

**Option 3: Mobile App**
- **Framework**: React Native or Flutter
- **Model**: TensorFlow Lite for mobile deployment

## Implementation Steps

### Phase 1: Model Preparation
1. Export the trained model from the Jupyter notebook
   - Save model as `.h5` or SavedModel format
   - Test model loading and prediction separately

2. Create a standalone prediction script
   - Function to preprocess images
   - Function to run inference
   - Function to post-process results (apply threshold)

### Phase 2: Backend Development
1. Set up Flask/FastAPI server
2. Create API endpoint for image upload
3. Implement image preprocessing pipeline
4. Load model and run inference
5. Return results as JSON

### Phase 3: Frontend Development
1. Create upload interface
2. Implement image preview
3. Connect to backend API
4. Display results dynamically
5. Add styling and responsive design

### Phase 4: Testing & Deployment
1. Test with various X-ray images
2. Validate predictions against known results
3. Optimize for performance
4. Deploy to hosting platform

## Example API Response Format

```json
{
  "success": true,
  "predictions": [
    {
      "condition_id": 1,
      "condition_name": "Condition Name",
      "probability": 0.95,
      "detected": true
    },
    {
      "condition_id": 2,
      "condition_name": "Another Condition",
      "probability": 0.92,
      "detected": true
    },
    // ... remaining conditions
  ],
  "detected_conditions": [
    "Condition Name",
    "Another Condition"
  ],
  "image_info": {
    "original_size": "512x512",
    "processed_size": "128x128"
  }
}
```

## Model Export Code

```python
# Save the trained model
model.save('arthritis_model.h5')

# Or save as SavedModel format
model.save('arthritis_model_saved')

# For TensorFlow.js (browser deployment)
# Install: pip install tensorflowjs
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, 'tfjs_model')
```

## Prediction Pipeline Code Template

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path):
    """Preprocess X-ray image for model input"""
    img = load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, 128, 128, 1)  # Add batch dimension
    return img_array

def predict_conditions(image_path, model, threshold=0.9):
    """Predict arthritis conditions from X-ray"""
    # Preprocess image
    processed_img = preprocess_image(image_path)

    # Get predictions
    predictions = model.predict(processed_img)[0]

    # Apply threshold
    detected = (predictions > threshold).astype(int)

    # Format results
    results = []
    for i, (prob, det) in enumerate(zip(predictions, detected)):
        results.append({
            'condition_id': i,
            'probability': float(prob),
            'detected': bool(det)
        })

    return results

# Load model
model = load_model('arthritis_model.h5')

# Make prediction
results = predict_conditions('xray_image.jpg', model)
```

## Important Notes

1. **Condition Names**: The actual names of the 11 conditions need to be extracted from the `_classes.csv` file in your dataset or from the Roboflow dataset documentation.

2. **Medical Disclaimer**: Since this is a medical diagnosis tool, include appropriate disclaimers:
   - "This tool is for educational/research purposes only"
   - "Not a substitute for professional medical diagnosis"
   - "Consult a healthcare professional for medical advice"

3. **Model Performance**:
   - Training accuracy: 85.2%
   - Test accuracy: 93.75%
   - Optimal threshold: 0.9

4. **Image Quality**: The model was trained on specific X-ray types from the Roboflow dataset. Performance may vary with different image sources.

## Next Steps

1. Decide on deployment platform (web/desktop/mobile)
2. Export the trained model from the notebook
3. Extract condition names from the dataset
4. Start with backend development (model serving)
5. Build frontend interface
6. Test and iterate

## Questions to Address

- What are the exact names of the 11 conditions being detected?
- Will this be a public-facing app or internal tool?
- Do you need user authentication or session management?
- Should the app store uploaded images and results?
- What are your deployment constraints (budget, hosting)?
