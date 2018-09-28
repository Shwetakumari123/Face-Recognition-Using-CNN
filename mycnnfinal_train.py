import cv2 
#It is used for resizing images
import numpy as np
#deals with array
import os

#dealing with directories
from random import shuffle
#randomizes the data of list
import tensorflow as tf

from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression



#make our loops show a smart progress meter 
TRAIN_DIR='/home/shweta/Documents/myfinalproject2.0/traindata'
IMG_SIZE=50 #size of image is 50*50 pixel
LR=1e-3 #This is learning rate here LR=0.001
#print(LR) 
MODEL_NAME='facedetection-{}-{}.model'.format(LR,'5conv-basic')# FOR printing name of the MODEL
#subject= ["", "Shweta", "Smriti"]
#print(MODEL_NAME) 
def label_img(img):
	word_label=img.split('.')[-3]
	if word_label=='shweta':return np.array([1,0,0,0,0,0,0])
	elif word_label=='smriti':return np.array([0,1,0,0,0,0,0])
	elif word_label=='Anu':return np.array([0,0,1,0,0,0,0])
	elif word_label=='Hari':return np.array([0,0,0,1,0,0,0])
	elif word_label=='bhagatsingh':return np.array([0,0,0,0,1,0,0])
	elif word_label=='Abdulkalam':return np.array([0,0,0,0,0,1,0])
	elif word_label=='megha':return np.array([0,0,0,0,0,0,1])
	




def create_train_data():
	training_data=[]
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label=label_img(img)
		path=os.path.join(TRAIN_DIR,img)
		img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
		training_data.append([np.array(img),np.array(label)])
	shuffle(training_data)
	np.save('train_data2.npy',training_data)
	return training_data

train_data=create_train_data()
tf.reset_default_graph()
#Deep learning library featuring a higher-level API for TensorFlow.

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 7, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
#if the model is not loaded then run the commented part
train = train_data[:-1]
test = train_data[-42:]
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]
model.fit({'input': X}, {'targets': Y}, n_epoch=50, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)


#model.load(model)


#for predicting the new image

	


