"""
Flask Web Application for Arthritis X-ray Classification
Allows users to upload X-ray images and get arthritis predictions
"""

import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from model_utils import build_model, preprocess_image, predict_arthritis
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Global variable to store loaded model
model = None


def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_trained_model(model_path='arthritis_model.h5'):
    """
    Load the trained model from file

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded Keras model
    """
    global model

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Model file not found at {model_path}")
        print("Building new model architecture (weights not trained)...")
        model = build_model()
        print("WARNING: Using untrained model. Please train and save the model first.")

    return model


@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return predictions"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIFF files.'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Reopen file for preprocessing
        with open(filepath, 'rb') as f:
            # Preprocess image
            processed_image = preprocess_image(f)

        # Make prediction
        results = predict_arthritis(model, processed_image)

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Load the trained model
    print("=" * 60)
    print("Arthritis X-ray Classification Web Application")
    print("=" * 60)
    load_trained_model()

    # Run the Flask app
    print("\nStarting web server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
