from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# dimensions of our images. if your raw data is not of this size,
# you will need to use ffmpeg (or some other tool) to figure out
#how to reformat your data
img_width, img_height = 256, 256

# this is where your training and testing data (images) need to be relatve to
# where your script (altimeter_nn.py) is
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape = input_shape, activation = 'relu'))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second and third convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])



epochs = 50
batch_size = 32

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
                                   rescale=1. / 255,
                                   rotation_range=180,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='wrap')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
                                                    train_data_dir,
                                                    color_mode=("grayscale"),
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
                                                        validation_data_dir,
                                                        color_mode=("grayscale"),
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

history = model.fit_generator(
                    train_generator,
                    steps_per_epoch = 2352 // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=1158 // batch_size)

print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model.png')

import matplotlib.pyplot as plt


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show(block=False)
import os

i = 0
while os.path.exists("C:/Users/Alec Otterson/AlecDocs/AI/DroneAlt/CNN/alt1Charts/acc%s.png" % i):
    i += 1

plt.savefig("C:/Users/Alec Otterson/AlecDocs/AI/DroneAlt/CNN/alt1Charts/acc%s.png" % i)

model.save_weights("C:/Users/Alec Otterson/AlecDocs/AI/DroneAlt/CNN/alt1Weights/weight%s.h5" % i)
