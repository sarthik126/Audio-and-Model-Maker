import librosa
import soundfile
import os, glob, pickle
import numpy as np
import sounddevice
from scipy.io.wavfile import write

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

modelname=input("Enter model name: ")
model=pickle.load(open("Models/"+modelname,"rb"))

fs = 16000
second = 3

print("Recording...")

while True:
    print(".................")
    record_voice = sounddevice.rec(int(second*fs),samplerate=fs,channels=1)
    sounddevice.wait()
    write("continous_mic_output.wav",fs,record_voice)
    x_test1=load_data("continous_mic_output.wav")
    y_pred1=model.predict(x_test1)
    print("Predicted Value: "+str(y_pred1))

