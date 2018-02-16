import os
import numpy as np
import pandas as pd
from skimage.io import imread 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

NUM_IMG = 25

def get_training_data(train_path, labels_path):
	print(labels_path)
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

	labels_df = pd.read_csv(labels_path) #["Finding Labels"]
	labels_df = labels_df["Finding Labels"]
	labels = np.zeros(NUM_IMG) # 0 for no finding, 1 for finding.

	# loading all labels
	for i in range(NUM_IMG):
		if (labels_df[i] == 'No Finding'):
			labels[i] = 0
		else:
			labels[i] = 1
	images = np.expand_dims(np.array(features), axis=3).astype('float32') / 255 # adding single channel
	print(images.shape)
	return images, labels
	
X_train, y_train = get_training_data("data/train/", "data/train-labels.csv")
X_test, y_test = get_training_data("data/test/", "data/test-labels.csv")
 
model = Sequential()

 
model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(1024, 1024, 1), data_format='channels_last'))
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
model.fit(X_train, y_train, batch_size=4, nb_epoch=10, verbose=1)
 
score = model.evaluate(X_test, y_test, verbose=0)
