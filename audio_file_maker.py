import sounddevice
from scipy.io.wavfile import write

def main_function():

    fs = 16000
    second = 5

    emotions={
        '1':'neutral',
        '2':'calm',
        '3':'happy',
        '4':'sad',
        '5':'angry',
        '6':'fearful',
        '7':'disgust',
        '8':'surprised',
        '9':'(Exit Loop)'
    }

    while True:
        for a in emotions:
            print(a+" : "+emotions[a])
        x = input("Enter the emotion number as above:")
        if x == '9':
            break
        file = "Audios/"+emotions[x]+".wav"
        print("Recording... ( "+emotions[x]+" )")

        record_voice = sounddevice.rec(int(second*fs),samplerate=fs,channels=1)
        sounddevice.wait()
        write(file,fs,record_voice)

        print("............................")
    return

main_function()