#VOCARM Project
#Feature extraction class
#Richard Masson

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def readData(path):
    # Read all the folders found in the path, and determine the class list based on this
    print("Reading data from directory", path)
    directory = os.path.join(os.path.dirname(__file__),path)
    folders = os.listdir(directory)
    return(folders)

def readWav(folder):
    # Retrieve the name with file exension of wav files in a folder
    folder = os.path.join(os.path.dirname(__file__),folder)
    paths = os.listdir(folder)
    wav_names = [f[:-4] for f in paths if f.endswith(".wav")]
    return(wav_names)

def getWavFiles(path):
    # Get the exact location of all wav files (ie data samples) in a given folder
    print("Retreiving wav files from directory", path, "...")
    folders = readData(path)
    wav_files = []
    for folder in folders:
        wav_names = readWav(path+"//"+folder)
        wav_files_folder = [os.path.join(folder,(f+".wav")) for f in wav_names]
        wav_files.append(wav_files_folder)
    print("Successfully found wav files.")
    wav_files = np.array(wav_files, dtype=object)
    return wav_files

def extractData(path):
    # Go through each folder and extract features to be added to a training set
    file_set = getWavFiles(path)
    y_list = []
    x_list = []
    diction = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'up':10,
        'down':11,
        'left':12,
        'right':13
    }
    ideal_shape = (128, 44) #Shape size everything must conform to
    for folder_set in file_set:
        for data in folder_set:
            y_string = data.split("//")[0]
            y_list.append(diction.get(y_string, y_string))
            appendix = getMelFeature(path+"//"+data)
            appendix = np.array(appendix)
            if appendix.shape != ideal_shape:
                if (appendix.shape[0] < 128 or appendix.shape[1] < 44):
                    shape = np.shape(appendix)
                    pad_array = np.zeros((128,44))
                    pad_array[:shape[0],:shape[1]] = appendix
                    x_list.append(pad_array)
                else:
                    shape = np.shape(appendix)
                    reduced_array = np.zeros((128,44))
                    reduced_array = appendix[:128,:44]
                    x_list.append(reduced_array)
            else:
                x_list.append(appendix)
    x_list = np.stack(x_list, 0)
    y_list = y_list
    print("Successfully built data set using", path)
    return y_list, x_list

def getMelFeature(filename):
    # Obtain the mel spectrogram(2-d) for a file
    y, sr = librosa.load(filename)
    spectrogram = np.array(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000))
    print("Extracted spectrogram from "+filename)
    return(spectrogram)

def processCommand(filename):
    y, sr = librosa.load(filename)
    spectro = np.array(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000))
    ideal_shape = (128, 44)
    output = np.array(spectro)
    if output.shape != ideal_shape:
        if (output.shape[0] < 128 or output.shape[1] < 44):
            shape = np.shape(output)
            pad_array = np.zeros((128,44))
            pad_array[:shape[0],:shape[1]] = output
            output = pad_array
        else:
            shape = np.shape(output)
            reduced_array = np.zeros((128,44))
            reduced_array = output[:128,:44]
            output = reduced_array
    output = np.expand_dims(output, axis=2)
    #output = np.array([output,])
    output = np.expand_dims(output, axis=0)
    return output
