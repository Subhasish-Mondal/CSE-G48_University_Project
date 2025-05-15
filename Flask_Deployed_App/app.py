import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from PIL import Image
import joblib

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# === Load models ===
sgd_model = joblib.load('./sgd_model.pkl')  # Trained SGD classifier
feature_extractor = load_model('./resnet50_feature_extractor.h5')  # Trained ResNet50 feature extractor

# === Load CSV data ===
plant_info = pd.read_csv('./medicinal_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('./supplement_info.csv', encoding='cp1252')

# === Preprocess image and extract features ===
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array, verbose=0)
    return features

# === Predict plant class ===
def prediction(image_path):
    features = preprocess_image(image_path)
    pred_label = sgd_model.predict(features)[0]  # e.g., 'Arali'
    return pred_label

# === Routes ===
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image_file = request.files['image']
        filename = image_file.filename
        file_path = os.path.join('./static/uploads', filename)
        image_file.save(file_path)

        pred = prediction(file_path)  # returns string label like 'Aloevera'

        # Fetch matching plant info
        try:
            plant_row = plant_info[plant_info['disease_name'].str.lower() == pred.lower()].iloc[0]
            plant_name = plant_row['disease_name']
            description = plant_row['description']
            usage = plant_row['Possible Steps']  
            image_url = plant_row['image_url']
        except IndexError:
            return f"❌ Plant info not found for prediction: {pred}", 404

        # Fetch matching supplement info (partial match)
        supplement_row = supplement_info[supplement_info['supplement_name'].str.contains(pred, case=False)]
        if not supplement_row.empty:
            supplement_row = supplement_row.iloc[0]
            supplement_name = supplement_row['supplement_name']
            supplement_image_url = supplement_row['supplement_image']
            supplement_buy_link = supplement_row['buy link']
        else:
            supplement_name = "Not available"
            supplement_image_url = "./static/uploads/placeholder.jpg" 
            supplement_buy_link = "#"

        return render_template('submit.html',
                               title=plant_name,
                               desc=description,
                               prevent=usage,
                               image_url=image_url,
                               pred=pred,
                               sname=supplement_name,
                               simage=supplement_image_url,
                               buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html',
                           supplement_image=list(supplement_info['supplement_image']),
                           supplement_name=list(supplement_info['supplement_name']),
                           disease=list(plant_info['disease_name']),
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
