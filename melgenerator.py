#VOCARM Project
#Mel spectrum figure generator
#Richard Masson

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

location = "output.wav"
y, sr = librosa.load(location)
spectro = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
spectro_dB = librosa.power_to_db(spectro, ref=np.max)

plt.figure()
librosa.display.specshow(spectro_dB, y_axis='mel', x_axis='time')
plt.colorbar()
plt.show()