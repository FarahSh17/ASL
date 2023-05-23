from flask import Flask, render_template, request
import numpy as np
from PIL import Image
# from keras.models import load_model
# import tensorflow
from utils import load_aslmodel
from utils import load_variable

MODEL_PATH = r'model/best_model.h5'
MAPPING_PATH = r'model/mapping.pkl'

mapping = load_variable(MAPPING_PATH)
# reversed_mapping = {value: key for key, value in mapping.items()}
model = load_aslmodel(MODEL_PATH)



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image = request.files['image']

    # Process the image
    img = Image.open(image)
    img = img.resize((256,256))  # Resize the image to match your model's input size
    img_array = np.array(img)
    # img_array = preprocess_image(img_array)  # Preprocess the image as required by your model

    # Pass the image to your model for classification
    prediction = model.predict(np.expand_dims(img_array, axis=0))

    # Perform post-processing on the prediction (e.g., decode class labels)

    # Return the predicted result to the user
    return "Predicted class:  " +  mapping[np.argmax(prediction)]

if __name__ == '__main__':
    app.run()
