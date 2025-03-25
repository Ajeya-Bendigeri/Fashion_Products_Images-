import os
import pickle
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)

# Load models
models = {
    "category": load_model("model_category.h5"),
    "subcategory": load_model("model_subcategory.h5"),
    "season": load_model("model_season.h5"),
    "gender": load_model("model_gender.h5")
}

# Load class mappings
key_lists = {
    "category": pickle.load(open("model_category_keys.pkl", "rb")),
    "subcategory": pickle.load(open("model_subcategory_keys.pkl", "rb")),
    "season": pickle.load(open("model_season_keys.pkl", "rb")),
    "gender": pickle.load(open("model_gender_keys.pkl", "rb"))
}

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(data_path):
    os.makedirs(data_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if not file:
        return "No file uploaded"
    
    filepath = os.path.join(data_path, file.filename)
    file.save(filepath)
    
    test_image = image.load_img(filepath, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    predictions = {}
    for key, model in models.items():
        result = model.predict(test_image)
        predicted_index = np.argmax(result)
        predictions[key] = key_lists[key][predicted_index]
    
    return render_template("result.html", image_name=file.filename, 
                           category=predictions["category"],
                           subcategory=predictions["subcategory"],
                           season=predictions["season"],
                           gender=predictions["gender"])

@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(data_path, filename)

if __name__ == "__main__":
    app.run(debug=False)
