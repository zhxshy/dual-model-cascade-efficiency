"""
Project: Optimization of Computer Vision via Dual-Model Cascade
Author: Zhoshy Khalelov
Description: Script for testing inference confidence levels of MobileNetV3 and EfficientNet on the Uralsk Urban Dataset.
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.applications import mobilenet_v3, efficientnet
    print("TensorFlow loaded successfully")
except ImportError:
    print("Error: Tensorflow not found. Install it via 'pip install tensorflow'")


model_light = tf.keras.applications.MobileNetV3Small(weights='imagenet')
model_heavy = tf.keras.applications.EfficientNetB0(weights='imagenet')

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    x = np.array(img).astype('float32')
    x = np.expand_dims(x, axis=0)
    
    x_light = tf.keras.applications.mobilenet_v3.preprocess_input(x.copy())
    preds_light = model_light.predict(x_light)

    decoded_light = tf.keras.applications.mobilenet_v3.decode_predictions(preds_light, top=1)[0][0]
    
    confidence = decoded_light[2]
    result = {
        "file": os.path.basename(img_path),
        "light_label": decoded_light[1],
        "light_conf": confidence,
        "used_heavy": False,
        "final_label": decoded_light[1]
    }


    if confidence < 0.85:
        x_heavy = tf.keras.applications.efficientnet.preprocess_input(x.copy())
        preds_heavy = model_heavy.predict(x_heavy)
        decoded_heavy = tf.keras.applications.efficientnet.decode_predictions(preds_heavy, top=1)[0][0]
        
        result["used_heavy"] = True
        result["final_label"] = decoded_heavy[1]
        result["heavy_conf"] = decoded_heavy[2]
    
    return result

folders = ["data/day", "data/evening"] 
stats = {
    "data/day": [],
    "data/evening": []
}

for folder in folders:
    print(f"\nProcessing: {folder} ---")
    if not os.path.exists(folder):
        print(f"Directory {folder} not found.")
        continue
        
    for img_name in os.listdir(folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                res = predict_image(os.path.join(folder, img_name))
                status = "HEAVY" if res['used_heavy'] else "LIGHT"
                print(f"{status} | File: {res['file']} | Result: {res['final_label']} (Conf: {res['light_conf']:.2f})")
            except Exception as e:
                print(f"Error processing {img_name}: {e}") 