# Arthritis X-ray Classification Web Application

A professional web interface for the arthritis X-ray classification model that allows users to upload X-ray images and receive AI-powered diagnosis predictions.

## Features

- 🎨 **Professional Medical Interface** - Clean, intuitive design optimized for medical use
- 📁 **Drag & Drop Upload** - Easy image upload with drag-and-drop support
- 🔍 **Real-time Analysis** - Instant predictions on uploaded X-ray images
- 📊 **Detailed Results** - Visual probability analysis for all 11 arthritis conditions
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile devices
- ⚠️ **Medical Disclaimer** - Clear educational/research use notice

## Model Performance

- **Test Accuracy**: 93.75%
- **Precision**: 99% (micro-average)
- **Recall**: 100% (micro-average)
- **Architecture**: 4-layer CNN with 2.75M parameters

## Prerequisites

Before running the web application, you need:

1. Python 3.8 or higher
2. Your trained model saved as a `.h5` file

## Step 1: Save Your Trained Model

First, you need to save your trained model from the Jupyter notebook. Add this code at the end of your `Image_analysis_multiclass_balanced.ipynb` notebook:

```python
# Save the trained model
model.save('arthritis_model.h5')
print("Model saved successfully as 'arthritis_model.h5'")
```

Then move the saved model file to the `webapp` directory:

```bash
# Move the model file to the webapp directory
mv arthritis_model.h5 /path/to/Project_Arthritis/webapp/
```

## Step 2: Install Dependencies

Navigate to the webapp directory and install the required packages:

```bash
cd webapp
pip install -r requirements.txt
```

### Requirements

The following packages will be installed:
- Flask 3.0.0 - Web framework
- TensorFlow 2.15.0 - Deep learning framework
- NumPy 1.24.3 - Numerical computing
- Pillow 10.1.0 - Image processing
- Werkzeug 3.0.1 - WSGI utilities

## Step 3: Run the Web Application

Start the Flask server:

```bash
python app.py
```

You should see output like:

```
============================================================
Arthritis X-ray Classification Web Application
============================================================
Loading model from arthritis_model.h5...
Model loaded successfully!

Starting web server...
Open your browser and navigate to: http://localhost:5000
============================================================
```

## Step 4: Use the Application

1. Open your web browser and go to `http://localhost:5000`
2. Upload an X-ray image by:
   - Clicking "Choose File" button, or
   - Dragging and dropping an image onto the upload area
3. Click "Analyze X-ray" button
4. View the results showing detected conditions and probability scores

## Project Structure

```
webapp/
├── app.py                      # Main Flask application
├── model_utils.py              # Model architecture and preprocessing utilities
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── arthritis_model.h5          # Trained model (you need to add this)
├── templates/
│   └── index.html              # Main web interface
├── static/
│   └── css/
│       └── style.css           # Styling
└── uploads/                    # Temporary storage for uploaded images
```

## Supported Image Formats

- PNG
- JPG/JPEG
- BMP
- TIFF

## Model Classes

The model predicts probabilities for 11 arthritis-related conditions:

1. Class 0
2. Class 1
3. Class 2
4. Class 3
5. Class 4
6. Class 5
7. Class 6
8. Class 7
9. Class 8
10. Class 9
11. Class 10

**Note**: To get the actual class names, check the `_classes.csv` file from your training data and update the `CLASS_LABELS` list in `model_utils.py`.

## Configuration

You can modify these settings in `app.py`:

```python
# Maximum file size (default: 16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed file extensions
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Server port (default: 5000)
app.run(debug=True, host='0.0.0.0', port=5000)
```

## Troubleshooting

### Model File Not Found

If you see `Model file not found`, ensure:
1. You've saved the model using `model.save('arthritis_model.h5')` in your notebook
2. The model file is in the `webapp` directory
3. The filename is exactly `arthritis_model.h5`

### Import Errors

If you get import errors:
```bash
pip install --upgrade -r requirements.txt
```

### Port Already in Use

If port 5000 is already in use, change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Use port 8080 instead
```

### Memory Issues

If you encounter memory issues with TensorFlow:
```python
# Add this at the top of app.py
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Use CPU only
```

## Security Notes

⚠️ **Important**: This is a development server configuration.

For production deployment:
1. Set `debug=False` in `app.py`
2. Use a production WSGI server (Gunicorn, uWSGI)
3. Add authentication for sensitive medical data
4. Use HTTPS for secure communication
5. Implement rate limiting
6. Add input validation and sanitization

## Medical Disclaimer

This tool is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

## License

This project is part of the Project Arthritis repository. Please refer to the main repository for licensing information.

## Support

For issues or questions:
1. Check this README
2. Review the main project documentation
3. Check the original notebook for model details

## Updates and Customization

### Updating Class Labels

To update the actual class names, modify `CLASS_LABELS` in `model_utils.py`:

```python
CLASS_LABELS = [
    "Arthritis Type 1",
    "Arthritis Type 2",
    # ... etc
]
```

### Adjusting Prediction Threshold

The default threshold is 0.9 (90%). To change it, modify the prediction call in `app.py`:

```python
results = predict_arthritis(model, processed_image, threshold=0.85)  # 85% threshold
```

### Customizing the Interface

- **Colors**: Edit `webapp/static/css/style.css`
- **Layout**: Edit `webapp/templates/index.html`
- **Logo**: Add your logo image to `webapp/static/` and reference it in the HTML

## API Endpoint

The application exposes a `/predict` endpoint that accepts POST requests:

```bash
curl -X POST -F "file=@xray_image.jpg" http://localhost:5000/predict
```

Response:
```json
{
  "success": true,
  "results": {
    "detected_conditions": [...],
    "all_predictions": [...]
  }
}
```

This can be integrated into other applications or automated workflows.
