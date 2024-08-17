from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import DenseNet201

# Load the pre-trained DenseNet201 model for feature extraction
fe = DenseNet201(weights='imagenet', include_top=False, pooling='avg')

app = Flask(__name__)
CORS(app)

# Load the pre-trained image captioning model
saved_caption_model = tf.keras.models.load_model("model.h5")

# Assuming you have saved the tokenizer and max_length used during training
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 34  # Replace with the actual max_length used during training

# Set the path for uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Image preprocessing function
def preprocess_image(image_path, img_size=224):
    img = load_img(image_path, color_mode='rgb', target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to extract features using a pre-trained model (e.g., DenseNet201)
def extract_features(image_path, model, img_size=224):
    img = preprocess_image(image_path, img_size)
    feature = model.predict(img, verbose=0)
    return feature


# Caption prediction function
def simple_predict_caption(model, image_features, tokenizer, max_length):
    in_text = "startseq"
    
    for _ in range(max_length):
        # Convert the current text into a sequence of integers
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        
        # Predict the next word
        y_pred = model.predict([image_features, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        
        # Convert the predicted integer back to a word
        word = tokenizer.index_word.get(y_pred, None)
        
        if word is None or word == 'endseq':
            break
            
        in_text += ' ' + word
    
    # Remove the start and end tokens from the final caption
    final_caption = in_text.replace('startseq ', '').replace(' endseq', '')
    
    return final_caption

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f'Image saved at: {file_path}')

        # Extract features from the uploaded image
        new_image_features = extract_features(file_path, fe)
        
        # Predict the caption for the uploaded image
        new_image_caption = simple_predict_caption(saved_caption_model, new_image_features, tokenizer, max_length)

        return jsonify({'message': 'Image uploaded successfully', 'file_path': file_path, 'caption': new_image_caption}), 200
    os.remove(file_path)
    return jsonify({'error': 'Failed to upload image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
