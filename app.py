from flask import Flask, render_template, request
import numpy as np
from keras.models import model_from_json
import os
import librosa
import tensorflow as tf
app = Flask(__name__)

# Load the trained model
json_file = open('\\EmotionDetect\\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("\\EmotionDetect\\Emotion_Voice_Detection_Model.h5")

# Define a function to extract features from the audio file
def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)
    return twodim

# Define a route for rendering the upload form
@app.route('/')
def upload_form():
    return render_template('\\EmotionDetect\\templates\\upload.html')

# Define a route for processing the uploaded file
@app.route('/', methods=['POST'])
def upload_file():
    # Get the uploaded file from the request
    file = request.files['audiofile']
    # Save the file to the server
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    # Extract features from the audio file
    features = extract_features(file_path)
    # Make a prediction using the trained model
    prediction = loaded_model.predict(features)
    # Get the predicted emotion label
    label = np.argmax(prediction)
    if label == 0:
        emotion = 'Angry'
    elif label == 1:
        emotion = 'Fearful'
    elif label == 2:
        emotion = 'Happy'
    elif label == 3:
        emotion = 'Neutral'
    elif label == 4:
        emotion = 'Sad'
    else:
        emotion = 'Surprised'
    # Render a template with the predicted emotion
    return render_template('\\EmotionDetect\\templates\\result.html', emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)
