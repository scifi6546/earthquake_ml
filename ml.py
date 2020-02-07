import tensorflow as tf
from keras.utils import to_categorical
import random

import numpy as np
import obspy as ob
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
import json
wf = np.load("wf_array.npy")
phase = np.load("phase_array.npy")

print(wf.shape)
wf_train = wf[:16000]
phase_train = phase[:16000]

wf_test = wf[16000:]
phase_test = phase[16000:]
model = model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units = 64,input_shape=(wf_train[0].shape)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(activation='softmax',units=2)
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
model.fit(wf_train,(phase_train),epochs=1)
print(model.evaluate(wf_test,phase_test))

