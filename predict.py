import cv2
import numpy as np
import os
from random import shuffle
import tensorflow as tf
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from PIL import Image, ImageDraw
import face_recognition
TEST_DIR='/home/shweta/Documents/myfinalproject2.0/testdata'
IMG_SIZE=50 #size of image is 50*50 pixel
LR=1e-3 #This is learning rate here LR=0.001
#print(LR) 
haar_face_cascade=cv2.CascadeClassifier('/home/shweta/OpenCV/data/haarcascades/haarcascade_frontalface_alt.xml')
MODEL_NAME='facedetection-{}-{}.model'.format(LR,'5conv-basic')# FOR printing name of the MODEL
def process_test_data():
	testing_data=[]
	for img in tqdm(os.listdir(TEST_DIR)):
		path=os.path.join(TEST_DIR,img)
		img_num=img.split('.')[0]
		img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		#img = haar_face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5);
		#img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img),img_num])
	
	shuffle(testing_data)
	np.save('test_data3.npy',testing_data)
	return testing_data

test_data=process_test_data()




#if the training data is already loaded then
train_data=np.load('train_data.npy')
#test_data=np.load('test_data.npy')
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
import matplotlib.pyplot as plt
fig=plt.figure()
for num,data in enumerate(test_data[:12]):
    
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(4,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])    
    if np.argmax(model_out) == 1: str_label='Smriti'
    elif np.argmax(model_out) == 2: str_label='Anu'
    elif np.argmax(model_out) == 3: str_label='Hari'	
    elif np.argmax(model_out) == 4: str_label='BhagatSingh'	
    elif np.argmax(model_out) == 5: str_label='AbdulKalam'
    elif np.argmax(model_out) == 6: str_label='megha'					
    else: str_label='Shweta'
    print("identified as",str_label)        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


	

from PIL import Image, ImageDraw
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("photo.jpg")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)
