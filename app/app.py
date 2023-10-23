from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Define the class labels
dic = {0: 'COVID', 1: 'Normal', 2: 'TB'}

# Load the trained model
model = load_model('mmodel.h5')

# Remove the following line as it is not required for TensorFlow 2.x
# model.make_predict_function()

# Function to predict the label of an image
def predict_label(img_path):
    # Load and preprocess the image
    img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
    img = tf.keras.utils.img_to_array(img) / 255.0
    x = np.expand_dims(img, axis=0)

    # Perform prediction
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions[0])
    predicted_label = dic[predicted_class]

    return predicted_label

# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe to Artificial Intelligence Hub..!!!"

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)

        # Call the prediction function
        prediction = predict_label(img_path)

        # Open a new window with the prediction result
        return render_template("result.html", prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.debug = True
    app.run()

