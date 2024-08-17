from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the saved model
model = tf.keras.models.load_model('cats_vs_dogs_model.h5')

# Function to preprocess the image
def preprocess_image(image_path, img_size=(64, 64)):
    # Load the image
    img = load_img(image_path, target_size=img_size)
    # Convert the image to an array
    img_array = img_to_array(img)
    # Normalize the image
    img_array = img_array / 255.0
    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def make_prediction(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    # Make prediction
    prediction = model.predict(processed_image)
    # Determine if the image is of a cat or a dog
    if prediction[0][0] > 0.5:
        result = 'Dog'
    else:
        result = 'Cat'
    return result

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Homepage route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = make_prediction(file_path)
            return render_template('index.html', result=result, image_url=filename)
    return render_template('index.html', result=None, image_url=None)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
