from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np

app = Flask(__name__)

dic = {0: 'Normal', 1: 'Doubtful', 2: 'Mild', 3: 'Moderate', 4: 'Severe'}

# Image Size
img_size = 256
model = load_model('model.h5')

def predict_label(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    i = image.img_to_array(resized) / 255.0
    i = i.reshape(1, img_size, img_size, 1)

    # Use predict method to get probabilities for each class
    probabilities = model.predict(i)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(probabilities, axis=1)[0]

    return dic[predicted_class_index]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe to Artificial Intelligence Hub..!!!"

@app.route("/predict", methods=['POST'])
def upload():
    if request.method == 'POST':
        img = request.files['file']
        img_path = "uploads/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        print(p)
        return str(p).lower()

if __name__ == '__main__':
    app.run(debug=True,port=9090)
