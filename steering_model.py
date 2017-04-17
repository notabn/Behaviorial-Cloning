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
from keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt
import keras.models as Model
import math
from keras.preprocessing.image import ImageDataGenerator


samples = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        #angle = float(line[3])
        #if angle == 0 and np.random.random() > 0.75:
        #    continue
        samples.append(line)



set_size= len(samples)

train_samples = samples[:math.floor(set_size*0.8)]
validation_samples =  samples[math.floor(set_size*0.8):]


col_size = 64
row_size = 64
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
    shape = image.shape
    cols = shape[1]
    rows = shape[0]
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang

def generator(samples,batch_size=32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        images = []
        angles = []
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(samples))
            batch_sample = samples[i_line]
            keep_pr = 0
            while keep_pr == 0:
                # Choose left / right / center image and compute new angle
                angle = float(batch_sample[3])
                img_choice = np.random.randint(3)
                if img_choice == 0:
                    img_path = './data/IMG/' + batch_sample[1].split('/')[-1]
                    angle += 0.25
                elif img_choice == 1:
                    img_path = './data/IMG/' + batch_sample[0].split('/')[-1]
                else:
                    img_path = './data/IMG/' + batch_sample[2].split('/')[-1]
                    angle -= 0.25
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image, angle = trans_image(image, angle, 120)
                shape = image.shape
                image = image[math.floor(shape[0] / 5):shape[0] - 25, 0:shape[1]]
                image = cv2.resize(image,(col_size,row_size),interpolation=cv2.INTER_AREA)
                ind_flip = np.random.randint(2)
                if ind_flip == 0:
                    image = cv2.flip(image, 1)
                    angle = -angle
                if abs(angle) < .1:
                    pr_val = np.random.uniform()
                    if pr_val < pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            images.append(image)
            angles.append(angle)

        X_train = np.array(images)
        y_train = np.array(angles)

        yield X_train,y_train




train_generator = generator(train_samples,batch_size=32)



validation_generator = generator(validation_samples,batch_size=32)





def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row_size,col_size,ch),output_shape=(row_size,col_size,ch)))
    model.add(Convolution2D(32, (3, 3), border_mode='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3), border_mode='valid',activation='elu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('elu'))
    model.add(Convolution2D(64, (3, 3), border_mode='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model


#stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')

#checkpoint = ModelCheckpoint('new_model.h5', monitor='val_loss',  verbose=0,save_best_only=True, mode='auto')

model = create_model()
model.compile(loss='mse', optimizer='adam')

pr_threshold = 1


for i in range(8):
    model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/32, validation_data=validation_generator,
                                 validation_steps=len(validation_samples)/32, epochs=1,verbose=1)

    pr_threshold = 1/(i+1)

model.save('model_max.h5')


