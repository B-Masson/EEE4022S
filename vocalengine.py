#VOCARM Project
#Engine class used to control command recording and predictions
#OS Version: Linux
#Richard Masson
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning) #Remove that list of deprecated numpy warnings (harmless)
from tensorflow import keras
import tensorflow as tf
#from playsound import playsound #NOT AVAILABLE ON LINUX
import recordaudioL as rec #L stands for Linux
import keyboard 
import time
import features as feat
import os

def engineSetup(path):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)
  # Path
  params_path = path
  # Load the model in
  print("Loading model...")
  model = keras.models.load_model(params_path+"//params")
  model.summary()
  return model

def predict(model):
    time.sleep(0.2)  
    recording = "output.wav"
    rec.record(recording)
    #rec.trim(recording) # Uncomment for recordings > 1s (see recordaudioL)
    print("Done recording.")
    tic = time.perf_counter()
    print("Analyzing input...")
    x_input = feat.processCommand("output.wav")
    prediction = model.predict_classes(x_input)
    toc = time.perf_counter()
    pretime = round((toc-tic),4)
    return prediction, pretime

# Function to test my curated self-recorded samples file
def test(model):
    print("Testing recorded user input...")
    path = "my_samples"
    directory = os.path.join(os.path.dirname(__file__),path)
    files = os.listdir(directory)
    for f in files:
      x_input = feat.processCommand(path+"//"+f)
      pred = model.predict_classes(x_input)
      ind = f.index('_')
      actual_class = f[ind+1:]
      if pred > 9:
        if pred == 10:
          pred = 'Up'
        elif pred == 11:
          pred = 'Down'
        elif pred == 12:
          pred = 'Left'
        elif pred == 13:
          pred = 'Right'
        else:
          pred = 'Error, unexpected class ID number.'
      print(actual_class,": Class ", pred, sep='')


