import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# read data from csv file
lines = []
with open('./new_data/driving_log.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
      
# create images and steering angle measurements array
images = []
measurements = []
for line in lines[1:]:
    # add center, left and right images to help car
    # stay close the center of the road
    for i in range(3):
        source_path = line[i]
        tokens = source_path.split('/')
        local_path = "./new_data/IMG/" + tokens[-1]
        image = cv2.imread(local_path)
        images.append(image)
    # modify steering angle for left and right images
    # by adding/subtracting correction to original steering
    # measurement. This helps the car steer towards the center
    # in case it gets to either edge of the road.
    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

# since Track 1 consists large part of steering towards the 
# right, augment the data set by flipping each image and adding
# the flipped image to the date set. Negate the steering angle
# for the flipped images and add this value to the measurements 
# array
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)
    
    
X_train = np.asarray(augmented_images)
y_train = np.asarray(augmented_measurements)

# Use Lenet architecture for the Model
model = Sequential()
# images are normalized 
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Top 70 pixels includes the sky and bottom 20 pixels includes hood of car
# Hence they are cropped from each image 
model.add(Cropping2D(cropping=((70,20), (0,0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
# Dropout layer added with keep_prob = 0.5 to reduce overfitting
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer="Adam", loss="mse")
# 10 epochs was found to be appropriate 
model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True)

model.save('model.h5')


    
    
    