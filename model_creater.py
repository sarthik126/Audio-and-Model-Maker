import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.neural_network import MLPClassifier

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust','neutral','sad','angry','surprised']

#DataFlair - Load the data and extract features for each sound file
def load_data():
    x,y=[],[]
    for file in observed_emotions:
        file_name="Audios/"+file+".wav"
        emotion=file
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file_name, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return np.array(x), y

#DataFlair - Split the dataset
print("Extracting Features...")
x_train,y_train=load_data()
print(x_train,y_train)
#DataFlair - Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#DataFlair - Train the model
print("Fitting the Model...")
model.fit(x_train,y_train)

filename=input("Enter model name:")
pickle.dump(model,open("Models/"+filename,"wb"))
