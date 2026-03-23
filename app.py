"""
Plant Disease Detection Web Application
Uses a pre-trained CNN model (MobileNetV2) fine-tuned on the PlantVillage dataset.
"""

import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Disease class labels (38 PlantVillage classes) ──────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

DISEASE_INFO = {
    'Apple___Apple_scab': {'treatment': 'Apply fungicides containing myclobutanil or captan. Remove infected leaves.', 'severity': 'Moderate'},
    'Apple___Black_rot': {'treatment': 'Prune infected branches. Apply copper-based fungicides.', 'severity': 'High'},
    'Apple___Cedar_apple_rust': {'treatment': 'Remove nearby cedar trees if possible. Apply fungicides in early spring.', 'severity': 'Moderate'},
    'Apple___healthy': {'treatment': 'No treatment needed. Maintain regular care.', 'severity': 'None'},
    'Blueberry___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
    'Cherry_(including_sour)___Powdery_mildew': {'treatment': 'Apply sulfur-based fungicides. Ensure good air circulation.', 'severity': 'Moderate'},
    'Cherry_(including_sour)___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'treatment': 'Use resistant varieties. Apply fungicides if severe.', 'severity': 'High'},
    'Corn_(maize)___Common_rust_': {'treatment': 'Plant resistant hybrids. Apply fungicides early.', 'severity': 'Moderate'},
    'Corn_(maize)___Northern_Leaf_Blight': {'treatment': 'Use resistant varieties. Rotate crops. Apply fungicides.', 'severity': 'High'},
    'Corn_(maize)___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
    'Grape___Black_rot': {'treatment': 'Remove mummified fruit. Apply mancozeb or myclobutanil fungicides.', 'severity': 'High'},
    'Grape___Esca_(Black_Measles)': {'treatment': 'Prune infected wood. Apply preventive fungicide treatments.', 'severity': 'High'},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'treatment': 'Apply copper-based fungicides. Improve drainage.', 'severity': 'Moderate'},
    'Grape___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
    'Orange___Haunglongbing_(Citrus_greening)': {'treatment': 'No cure available. Remove infected trees. Control psyllid vectors.', 'severity': 'Critical'},
    'Peach___Bacterial_spot': {'treatment': 'Apply copper-based bactericides. Choose resistant varieties.', 'severity': 'Moderate'},
    'Peach___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
    'Pepper,_bell___Bacterial_spot': {'treatment': 'Apply copper bactericides. Avoid overhead irrigation.', 'severity': 'Moderate'},
    'Pepper,_bell___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
    'Potato___Early_blight': {'treatment': 'Apply chlorothalonil or mancozeb fungicides. Rotate crops.', 'severity': 'Moderate'},
    'Potato___Late_blight': {'treatment': 'Apply metalaxyl-based fungicides immediately. Remove infected plants.', 'severity': 'Critical'},
    'Potato___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
    'Raspberry___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
    'Soybean___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
    'Squash___Powdery_mildew': {'treatment': 'Apply potassium bicarbonate or sulfur fungicides.', 'severity': 'Moderate'},
    'Strawberry___Leaf_scorch': {'treatment': 'Remove infected leaves. Apply fungicides. Improve air circulation.', 'severity': 'Moderate'},
    'Strawberry___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
    'Tomato___Bacterial_spot': {'treatment': 'Apply copper-based bactericides. Use disease-free seeds.', 'severity': 'Moderate'},
    'Tomato___Early_blight': {'treatment': 'Apply chlorothalonil fungicide. Remove lower infected leaves.', 'severity': 'Moderate'},
    'Tomato___Late_blight': {'treatment': 'Apply mefenoxam-based fungicides. Destroy infected plants.', 'severity': 'Critical'},
    'Tomato___Leaf_Mold': {'treatment': 'Improve greenhouse ventilation. Apply fungicides.', 'severity': 'Moderate'},
    'Tomato___Septoria_leaf_spot': {'treatment': 'Apply fungicides. Remove infected lower leaves. Mulch around plants.', 'severity': 'Moderate'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'treatment': 'Apply miticides or neem oil. Increase humidity.', 'severity': 'Moderate'},
    'Tomato___Target_Spot': {'treatment': 'Apply azoxystrobin or chlorothalonil. Rotate crops.', 'severity': 'Moderate'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'treatment': 'Control whitefly vectors. Remove infected plants. Use resistant varieties.', 'severity': 'High'},
    'Tomato___Tomato_mosaic_virus': {'treatment': 'Remove infected plants. Disinfect tools. Control aphid vectors.', 'severity': 'High'},
    'Tomato___healthy': {'treatment': 'No treatment needed.', 'severity': 'None'},
}

# ── Model Loading ────────────────────────────────────────────────────────────
model = None

def load_model():
    global model
    model_path = 'model/plant_disease_model.keras'
    if os.path.exists(model_path):
        print(f"[INFO] Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("[INFO] Model loaded successfully.")
    else:
        print("[WARN] No saved model found. Run 'python scripts/train_model.py' first.")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed_extensions:
        return jsonify({'error': f'Unsupported file type: .{ext}'}), 400

    image_bytes = file.read()
    img_array = preprocess_image(image_bytes)
    predictions = model.predict(img_array)[0]

    top_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = []
    for idx in top_indices:
        class_name = CLASS_NAMES[idx]
        confidence = float(predictions[idx]) * 100
        plant, *disease_parts = class_name.split('___')
        disease = disease_parts[0].replace('_', ' ') if disease_parts else 'Unknown'
        info = DISEASE_INFO.get(class_name, {'treatment': 'Consult an agronomist.', 'severity': 'Unknown'})
        top_predictions.append({
            'class': class_name,
            'plant': plant.replace('_', ' '),
            'disease': disease,
            'confidence': round(confidence, 2),
            'treatment': info['treatment'],
            'severity': info['severity']
        })

    # Encode image for display
    img_b64 = base64.b64encode(image_bytes).decode('utf-8')
    img_data_url = f"data:image/{ext};base64,{img_b64}"

    return jsonify({
        'predictions': top_predictions,
        'image': img_data_url
    })

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
