import csv
import cv2
import os
import numpy as np
import sklearn
import sklearn.utils
import math

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
                # Choose left / right / center image and compute new angle
                angle = float(batch_sample[3])
                img_choice = np.random.randint(3)
                if img_choice == 0:
                    img_path = './data/IMG/' + batch_sample[1].split('/')[-1]
                    angle += 0.2
                elif img_choice == 1:
                    img_path = './data/IMG/' + batch_sample[0].split('/')[-1]
                else:
                    img_path = './data/IMG/' + batch_sample[2].split('/')[-1]
                    angle -= 0.2
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                shape = image.shape
#                image = image[math.floor(shape[0] / 5):shape[0] - 25, 0:shape[1]]
                image = cv2.resize(image,(col_size,row_size),interpolation=cv2.INTER_AREA)
                ind_flip = np.random.randint(2)
                if ind_flip == 0:
                    image = cv2.flip(image, 1)
                    angle = -angle

                images.extend([image])
                angles.extend([angle])

        X_train = np.array(images)
        y_train = np.array(angles)

        yield sklearn.utils.shuffle(X_train,y_train)




train_generator = generator(train_samples,batch_size=32)
validation_generator = generator(validation_samples,batch_size=32)




model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row,col,ch),output_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((30,10), (0,0))))
#model.add(Convolution2D(3, kernel_size =(3, 3), padding='same'))
model.add(Convolution2D(32, kernel_size =(5, 5), strides =(2, 2), padding='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size =(5, 5), strides =(2, 2), padding='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(128, kernel_size =( 3, 3)))
model.add(Dropout(0.5))
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

#model = Model.load_model('model.h5')
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