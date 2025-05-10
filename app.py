import os
import shutil
from flask import Flask, render_template, request
import cv2
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cnn', methods=['GET', 'POST'])
def cnn():
    if request.method == 'POST':
        try:
            dirPath = "static/images"
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # Clear old images
            for fileName in os.listdir(dirPath):
                os.remove(os.path.join(dirPath, fileName))

            fileName = request.form['filename']
            src = os.path.join("test", fileName)
            dst = os.path.join(dirPath, fileName)

            if not os.path.exists(src):
                return render_template('index.html', cnn_accuracy="Image not found!")

            shutil.copy(src, dst)

            IMG_SIZE = 50
            LR = 1e-3
            MODEL_NAME = 'Braintumor-{}-{}.model'.format(LR, '2conv-basic')

            def process_verify_data():
                verifying_data = []
                for img in os.listdir(dirPath):
                    path = os.path.join(dirPath, img)
                    img_data = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img_data is not None:
                        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                        verifying_data.append([img_data, img])
                return verifying_data

            verify_data = process_verify_data()

            tf.compat.v1.reset_default_graph()

            convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
            convnet = conv_2d(convnet, 32, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)
            convnet = conv_2d(convnet, 64, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)
            convnet = conv_2d(convnet, 128, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)
            convnet = conv_2d(convnet, 32, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)
            convnet = conv_2d(convnet, 64, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)
            convnet = fully_connected(convnet, 1024, activation='relu')
            convnet = dropout(convnet, 0.8)
            convnet = fully_connected(convnet, 2, activation='softmax')
            convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                                 loss='categorical_crossentropy', name='targets')

            model = tflearn.DNN(convnet)

            if os.path.exists(f"{MODEL_NAME}.index"):
                model.load(MODEL_NAME)
            else:
                return render_template('index.html', cnn_accuracy="Model not found!")

            for img_data, img_name in verify_data:
                data = img_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
                prediction = model.predict(data)[0]
                predicted_class = np.argmax(prediction)
                confidence = prediction[predicted_class] * 100

                cnn_label = 'No Tumor' if predicted_class == 0 else 'Tumor'
                cnn_accuracy = f"The predicted image is {cnn_label} with an accuracy of {confidence:.2f}%"

                return render_template('index.html',
                                       cnn_label=cnn_label,
                                       cnn_accuracy=cnn_accuracy,
                                       ImageDisplay=f"/static/images/{fileName}")

        except Exception as e:
            return render_template('index.html', cnn_accuracy=f"Error: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
