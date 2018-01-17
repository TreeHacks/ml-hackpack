
import os
import numpy as np
import pandas as pd
from skimage.io import imread 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
NUM_IMG = 10
NUM_LABELS = 15 
def get_training_data(train_path):
	train_images = []
	train_files = []
	for filename in os.listdir(train_path):
		if filename.endswith(".png"):
			train_files.append(train_path + filename)
	n_examples = 0
	features = []
		
	for i, train_file in enumerate(train_files):
			if i >= NUM_IMG: break
			train_image = imread(train_file, True)
			feature_set = np.asarray(train_image)
			features.append(feature_set)
			n_examples += 1
	labels_df = pd.read_csv("labels.csv")
	labels = np.zeros((NUM_IMG, NUM_LABELS))
	cols = ["Finding Labels"]
	labels_df = labels_df[cols]
	given_labels = np.array(['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration','Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'])
	set_label = False
	for index, label in labels_df.iterrows():
		if index >= NUM_IMG: break
		findings = label['Finding Labels'].strip().split('|')
		labels[index - 1, given_labels.searchsorted(findings)] = 1
		
	return np.array(features), np.array(labels)
	
(X_train, y_train) = get_training_data("data/images_001/images/")
(X_test, y_test) = get_training_data("data/images_012_test/images/")

X_train = X_train.reshape(X_train.shape[0], 1, 1024, 1024)
X_test = X_test.reshape(X_test.shape[0], 1, 1024, 1024)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
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