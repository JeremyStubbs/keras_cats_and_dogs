#import modules
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
import os
import shutil
import random
import glob
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline

# Create train, validation and test subsets
os.chdir('data/dogs-vs-cats')
if os.path.isdir('train/dog') is False:
  os.makedir('train/dog')
  os.makedir('train/cat')
  os.makedir('valid/dog')
  os.makedir('valid/cat')
  os.makedir('test/dog')
  os.makedir('test/cat')

  for c in random.sample(glob.glob('cat*'), 500):
    shutil.move(c, 'train/cat')
  for c in random.sample(glob.glob('dog*'), 500):
    shutil.move(c, 'train/dog')
  for c in random.sample(glob.glob('cat*'), 100):
    shutil.move(c, 'valid/cat')
  for c in random.sample(glob.glob('dog*'), 100):
    shutil.move(c, 'valid/dog')
  for c in random.sample(glob.glob('cat*'), 50):
    shutil.move(c, 'test/cat')
  for c in random.sample(glob.glob('dog*'), 50):
    shutil.move(c, 'test/dog')

os.chdir('../../')

#set paths 
train_path = ''
valid_path = ''
test_path = ''

#Create generators for neural network
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat','dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat','dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, classes=None, target_size=(224,224), batch_size=10, shuffle=False)

#view images
def plotImages(images_arr):
  fig, axes = plt.subplots(1, 10, figsize = (20,20))
  axes = axes.flatten()
  for img, ax in zip(images_arr, axes):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout()
  plt.show()

plotImages(img)
print(labels)

#Import the pre-trained model
vgg16_model = tf.keras.applications.vgg16.VGG16()

#View summary
vgg16_model.summary()

# Create sequential model with layers set to layers of vgg16
model = keras.Sequential()
for layer in vgg16_model.layers[:-1]:
  model.add(layer)

# Prevent alteration of model
for layer in model.layers:
  layer.trainable = False

# add output layer
model.add(layers.Dense(units=2, activation = 'softmax'))

# compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# fit model
model.fit(
        train_batches,
        steps_per_epoch=100,
        epochs=5,
        validation_data=valid_batches,
        validation_steps=20)

# Predict images in test set with model 
predictions = model.predict(test_batches, verbose=0)
predictions

#View true values
test_batches.classes

#Convert predictions to 1D array
x = []
for i in range(len(predictions)):
  if predictions[i][0]>=predictions[i][1]:
    x.append(0)
  else:
    x.append(1)
answers = np.array(x)
print(answers)

#Plot confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test_batches.classes, answers)
disp = ConfusionMatrixDisplay(cm)
disp.plot()