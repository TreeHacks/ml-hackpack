
import os
import numpy as np
import pandas as pd
from skimage.io import imread 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

NUM_IMG = 50

def get_training_data(train_path):
	train_images = []
	train_files = []
	for filename in os.listdir(train_path):
		if filename.endswith(".png"):
			train_files.append(train_path + filename)

	features = []
		
	for i, train_file in enumerate(train_files):
			if i >= NUM_IMG: break
			train_image = imread(train_file, True)
			feature_set = np.asarray(train_image)
			features.append(feature_set)
			n_examples += 1

	labels_df = pd.read_csv("labels.csv")["Finding Labels"]
	labels = np.zeros(NUM_IMG) # 0 for no finding, 1 for finding.

	# loading all labels
	for i in range(NUM_IMG):
		if (labels_df[i] == 'No Finding'):
			labels[i] = 0
		else:
			labels[i] = 1
	print(features.shape)
	images = np.expand_dims(np.array(features), axis=3).astype('float32') / 255 # adding single channel
	return np.array(features), labels
	
X_train, y_train = get_training_data("data/train/")
X_test, y_test = get_training_data("data/test/")
 
#Y_train = np_utils.to_categorical(y_train, 14)
#Y_test = np_utils.to_categorical(y_test, 14)
 
model = Sequential()
 
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 1024, 1024), data_format='channels_first'))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
model.fit(X_train, y_train, batch_size=32, nb_epoch=10, verbose=1)
 
score = model.evaluate(X_test, y_test, verbose=0)