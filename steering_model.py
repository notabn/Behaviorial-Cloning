import csv
import cv2
import os
import numpy as np
import sklearn
import sklearn.utils

from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout,Activation
from keras.layers import Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import keras.models as Model

from keras.preprocessing.image import ImageDataGenerator


samples = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

del samples[0]

train_samples, validation_samples = train_test_split(samples,test_size=0.2)

col = 160
row = 80
ch = 3

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image, steer, trans_range):
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (col, row))

    return image_tr, steer_ang

def generator(samples,batch_size=32):
    num_samples = len(samples)
    col_size = 160
    row_size = 80
    max_zero_steering = 0.3*num_samples
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            zero_steering = 0
            for batch_sample in batch_samples:

                center_angle = float(batch_sample[3])
                # remove 70% from data when the car in driving straight ahead in order to prevent bias for driving straigth
                if ((zero_steering < max_zero_steering) and abs(center_angle) < 0.01) or abs(center_angle) > 0.01:
                    name = './data/IMG/' + batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    center_image = cv2.resize(center_image,(col_size,row_size),interpolation=cv2.INTER_AREA)
                    fliped_image = cv2.flip(center_image,1)
                    zero_steering += 1

                    # create new translated image
                    image_trans, y_steer = trans_image(center_image, center_angle, 100)

                    # create adjusted steering measurements for the side camera images
                    correction = 0.15
                    steering_left = center_angle + correction
                    steering_right = center_angle - correction

                    img_left = cv2.imread('./data/IMG/' + batch_sample[1].split('/')[-1])
                    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                    img_left = cv2.resize(img_left, (col_size, row_size), interpolation=cv2.INTER_AREA)
                    img_right = cv2.imread('./data/IMG/' + batch_sample[2].split('/')[-1])
                    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                    img_right = cv2.resize(img_right, (col_size, row_size), interpolation=cv2.INTER_AREA)
                    images.extend([center_image,image_trans,fliped_image,img_left,img_right])
                    angles.extend([center_angle,y_steer,center_angle*-1.0,steering_left,steering_right])

        X_train = np.array(images)
        y_train = np.array(angles)

        yield sklearn.utils.shuffle(X_train,y_train)




train_generator = generator(train_samples,batch_size=32)
validation_generator = generator(validation_samples,batch_size=32)




model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row,col,ch),output_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((int(1/5*col),25), (0,0))))
model.add(Convolution2D(32, kernel_size =(5, 5), strides =(2, 2), padding='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size =(5, 5), strides =(2, 2), padding='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(128, kernel_size =( 3, 3)))
model.add(Activation('relu'))
# 64@3x13
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))


checkpoint = ModelCheckpoint('model.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

model = Model.load_model('model.h5')
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/32, validation_data=validation_generator,
                                     validation_steps=len(validation_samples)/32, epochs=2,callbacks=[checkpoint],verbose=1)



model.save('model.h5')

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()