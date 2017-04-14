import csv
import cv2
import numpy as np




from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator


lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)




images = []
measurements = []

for i,line in enumerate(lines):
    if line[3] == 'steering':
        continue
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' +filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)


measurements = np.array(measurements)
images_flipped = np.fliplr(images)
measurements_flipped = -measurements

images = np.concatenate((images, images_flipped))
measurements = np.concatenate((measurements,measurements_flipped))

X_train = images
y_train = measurements

print(X_train.shape)

new_height = 160
new_width = 80
ch = 3

def preprocess(image):
    import tensorflow as tf
    resized = tf.image.resize_images(image, (new_height, new_width), method=0)
    normalized = resized/255.0 - 0.5
    return normalized

#datagen = ImageDataGenerator(rescale=1./2)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
#datagen.fit(X_train)


model = Sequential()
model.add(Lambda(lambda x: preprocess(x), input_shape=(160,320, 3), output_shape=(new_width,new_height,3)))
model.add(Convolution2D(32, 3, 3, border_mode='valid',activation='relu' ))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
#model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), samples_per_epoch=len(X_train))
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)


model.save('model.h5')