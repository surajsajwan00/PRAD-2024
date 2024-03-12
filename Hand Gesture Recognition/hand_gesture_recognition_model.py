## Libraries
import numpy as np
import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers
from keras.optimizers import Adam

## Load data
train_dir = "C:/iamworking/COLLEGE/YEAR-3/SEM-6/Pattern Recognition & Anomaly Detection LAB/PRAD-2024/PRAD project 1/CODE 2/dataset/train"
val_dir = "C:/iamworking/COLLEGE/YEAR-3/SEM-6/Pattern Recognition & Anomaly Detection LAB/PRAD-2024/PRAD project 1/CODE 2/dataset/validation"
test_dir = "C:/iamworking/COLLEGE/YEAR-3/SEM-6/Pattern Recognition & Anomaly Detection LAB/PRAD-2024/PRAD project 1/CODE 2/dataset/test"

outputSize = len(os.listdir(train_dir)) 
epochs = 30

## Data Augmentation
# Train Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Test Data Augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (256,256),
    batch_size = 32,
    class_mode = 'categorical',
    color_mode = 'grayscale'
)

val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size = (256,256),
    batch_size = 32,
    class_mode = 'categorical',
    color_mode = 'grayscale'
)


## CNN model function 
def create_model(outputSize):
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (256,256,1)))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Flatten())
    model.add(Dropout(rate = 0.5))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(units = outputSize, activation = 'softmax'))
    model.compile(optimizer = Adam(lr=1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


## Model creation
model = create_model(outputSize)

## Model  summary
model.summary()

## Fitting the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=outputSize*1000/32,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=outputSize*500/32
)

## Plotting Model's training Accuracy and Loss for training and validation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
range_ep = epochs +1
epoch_x = range(1, range_ep)

plt.plot(epoch_x,acc,'bo',label="Training Acc")
plt.plot(epoch_x,val_acc,'b',label='Validation Acc')
plt.title('Training and Validation Acc')
plt.legend()
plt.figure()

plt.plot(epoch_x,loss,'bo',label="Training Loss")
plt.plot(epoch_x,val_loss,'b',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.figure()

plt.show()

## Test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (256,256),
    batch_size = 32,
    class_mode = 'categorical',
    color_mode = 'grayscale'
)

## Model perfromance 
# Test accuracy and loss
test_loss, test_acc = model.evaluate_generator(test_generator,steps = outputSize*500/32)
print("Test Acc:",test_acc)
print("Test Loss:",test_loss)

## Save the model
model.save("gesture_model.h5")