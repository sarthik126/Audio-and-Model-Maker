# Audio-and-Model-Maker
Speech Emotion Recognition - Audio Feature Extractor with File Generator and Model Creator.

Requirements Installation:
  1. Move to root folder of the project in the command prompt.
  2. Run the followig command : $ pip install -r requirements.txt
  3. All the project necessary libraries will be installed.

Running Project:
  1. Move to root folder of the project in the command prompt.
  2. Run the python files as per requirements.

Note:
  1. The custom audio files and models can be made and can be linked with the main project (Add your custom model as model in static folder).
  2. Python file and its usage:
      1. audio_file_maker.py - For creating audio files
      2. model_creater.py    - For creating model out of audio
      3. file_prediction.py  - For predicting .wav audio files with created model (file name should be given as .wav and file should be present in Audios Directory)
      4. mic_prediction.py   - For continuous prediction of microphone audio with created model
  3. Running python files command : $python filename
  4. "Note that audio must be created for all types of emotions and this code will allow one audio for each emotion type and if same emotion audio file is created then old file will be replaced".

Project Description:
  1. This project is extension of Speech Emotion Recognition from Microphone project ( Main Project Link : https://github.com/sarthik126/Speech-Emotion-Recognition ).
  2. This project is used for creating .wav audio files and to create model out of created audio files.
  3. Accuracy can be increased using best model created using proper and more audios.
  4. This project is Speech Emotion Recognition without django integration and can be used for testing purpose.
