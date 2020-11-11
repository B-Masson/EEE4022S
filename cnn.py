#VOCARM Project
#Convolutional Neural Network
#Richard Masson

import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import features as feat
import numpy as np
from keras.utils import to_categorical
import time
import datetime
from matplotlib import pyplot as plt
import pandas as pd

print("Starting up the CNN")
# Relevant GPU allocation stuff for Tensorflow
# Code obtained from Tensorflow website on managing GPU resources
# https://www.tensorflow.org/guide/gpu
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

# Set meta parameters here (things that depend on the data being used)
input_size = 100
output_size = 14 #Number of classes. Currently: 9 numbers, 4 directions and 1 other

# Validation flag
validate = False

# Set path strings here
train_path = "train_full"
params_path = "Parameters"
params_file = "params"

# Block for testing purposes while I get this to work
print("Training path is:", train_path)
classes = feat.readData(train_path)
class_no = len(classes)
print("No of classes:", class_no)

#Start processing timer
tic = time.perf_counter()

# Populate data storage
y_train, x_train = feat.extractData(train_path)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, stratify=y_train, test_size=0.3) #Change based on data size

# Refine data array shapes
x_train = x_train.reshape(x_train.shape[0], 128, 44, 1)
x_test = x_test.reshape(x_test.shape[0], 128, 44, 1)
y_train = np.asarray(y_train).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)
y_train = to_categorical(y_train)
y_test= to_categorical(y_test)
toc = time.perf_counter() #End pre-processing timer

# Testing stuff
print("x count:", len(x_train))
print("test count:", len(x_test))
print("x[0] shape:", x_train[0].shape)
print("y count:", len(y_train))
print("y[0] shape:", y_train[0].shape)
time_data = round((toc-tic),4)

# Start training timer
tic = time.perf_counter()

# Current CNN model parameters
# Currently using parameters from an old project as a placeholder
input_shape = x_train[0].shape
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128, 44, 1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) #Kernal size determines filter matrix
model.add(MaxPooling2D(pool_size=(2, 2))) #Down-sample/reduce dimensionality using Max function
model.add(Dropout(0.5)) #Randomly ignore some nodes per pass. Reduces overfitting
model.add(Flatten()) #Connects previous layers to dense layer
model.add(Dense(128, activation='relu')) #Output layer to solve pattern #kernel_regularizer=keras.regularizers.l2(l=0.1)
model.add(Dropout(0.2)) #More overfitting reduction
model.add(Dense(14, activation='softmax')) #Ensures model outputs class. 14 = no. of classes and softmax ensures binary classes

opt = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
# Compile model for crossentropy-type loss and accuracy evaluation
model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

# Fit training data to our model
epochnum = 9 # Increase if validation testing is being conducted
bsize = 20 #Can be increased if computer can handle the memory cost
if (validate == True):
  early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
  ca = ModelCheckpoint('optimal_model_acc.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
  cl = ModelCheckpoint('optimal_model_loss.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
  x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)
  history = model.fit(x_train, y_train, validation_data=(x_vali,y_vali), epochs=epochnum, batch_size=20, callbacks=[early_stop, ca, cl]) #Ideal: 100 but change if memory issues prevail
else:
  history = model.fit(x_train, y_train, epochs=epochnum, batch_size=bsize)
toc = time.perf_counter() #End timer
time_train = round((toc-tic),4)
print("Training complete. Running test...")

# Start testing timer
tic = time.perf_counter()

# Evaluate accuracy based on test data
_, accuracy = model.evaluate(x_test, y_test, verbose=1)
y_labels =np.argmax(y_test, axis=1)
y_pred = model.predict_classes(x_test)
conf = pd.DataFrame(
    confusion_matrix(y_labels, y_pred),
    index=['true:1', 'true:2', 'true:3', 'true:4', 'true:5', 'true:6', 'true:7', 'true:8', 'true:9', 'true:up', 'true:down', 'true:left', 'true:right'],
    columns=['pred:1', 'pred:2', 'pred:3', 'pred:4', 'pred:5', 'pred:6', 'pred:7', 'pred:8', 'pred:9', 'pred:up', 'pred:down', 'pred:left', 'pred:right'])
print(conf)
print('Tested Accuracy: %.2f' % (accuracy*100))
if (validate == True):
  optmodel_a = load_model('optimal_model_acc.h5')
  _, accuracy_opt = optmodel_a.evaluate(x_test, y_test, verbose=1)
  print('Optimised (accuracy): %.2f' % (accuracy_opt*100))
  optmodel_l = load_model('optimal_model_loss.h5')
  _, loss_opt = optmodel_l.evaluate(x_test, y_test, verbose=1)
  print('Optimised (loss): %.2f' % (loss_opt*100))
toc = time.perf_counter() #End timer

if (validate == True):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

time_test = round((toc-tic),4)
time_total = time_data+time_train+time_test
print("Timing Information:\nPre-processing: ", time_data, "s\nTraining: ", time_train, "s\nTesting: ", time_test, "s", sep='')
choice = input("Would you like to save these paramters? (y)es / (n)o / (c)change file name and send to backup?\n")
# Save parameters
if (choice == 'y' or choice == 'c'):
  if (choice == 'c'):
    params_path = "Parameter_Backups" #Hard coding for now since I always use the same alt. folder
    params_file = input("Enter parameter file name:\n")
  model.save(params_path+"\\"+params_file)
  f = open(params_path+"\\"+params_file+"_info.txt", "w")
  text = "Parameter information:\nFolder used: %s\nAccuracy: %.2f\nTotal time: %.2f\nLogged: " % (train_path, accuracy*100, time_total)
  now = datetime.datetime.now()
  text += now.strftime("%Y-%m-%d %H:%M:%S")
  f.write(text)
  f.close()
  c = open("graphdata.csv", 'a+')
  csvtext = "%s,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (params_file, accuracy*100, time_data, time_train, time_test, time_total)
  c.write(csvtext)
  c.close()
  print("Saved params.")
elif (choice == 'n'):
  print("Understood.")
else:
  print("I'll take that as a no.")

'''
# Training metrics
secondary = input("Would you like to save training metrics? (y/n)\n")
if (secondary == 'y'):
  loss = hist.history['loss']
  acc = hist.history['accuracy']
  m = open("Graphing"+"\\"+train_path+"_metrics.csv", "w")
  metrictext = "Epoch,Loss,Accuracy"
  for i in range(epochnum):
    metrictext += "\n%d,%.4f,%.4f" % (i+1, loss[i], acc[i])
  m.write(metrictext)
  m.close()
  print("Wrote metrics.")
print("Closing program...")
'''