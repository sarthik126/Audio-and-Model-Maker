import librosa
import soundfile
import os, glob, pickle
import numpy as np

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature1(file_name, mfcc, chroma, mel):
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


# DataFlair - Load the data and extract features for each sound file
def load_data(filename):
    x = []
    for file in glob.glob(filename):
        file_name = os.path.basename(file)
        
        #sound = AudioSegment.from_mp3(filename)
        #sound.export(file, format="wav")

        feature = extract_feature1(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
    return x

filename=input("Enter the model name:")
model=pickle.load(open("Models/"+filename,"rb"))

filename=input("Enter filename to predict:")
x_test=load_data("Audios/"+filename)

y_pred=model.predict(x_test)
print(y_pred)