from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import os
import random

### You may need to adjust these paths for ###
### your own operating system.             ###
data_dir = "data"
validation_dir = "validation"

def create_validation_data(howmany):
  """
  Create validation data by copying images from the data directory to the validation 
  directory. It only copies if a random number generated between 1 and `howmany`
  is equal to 1, so if you pass in 10, roughly 1/10 of the  images will be used for 
  validation. 
  Returns a tuple of the number of training images ad the number of validation images.
  """
  datasets = os.listdir(data_dir)
  nval = 0
  ndat = 0
  for s in datasets:
    os.makedirs(os.path.join(validation_dir, s))
    images = os.listdir(os.path.join(data_dir, s))
    for i in images:
      if random.randint(1, howmany) == 1:
        os.rename(os.path.join(data_dir, s, i), os.path.join(validation_dir, s, i))
        nval += 1
      else:
        ndat += 1
  return ndat, nval



def remove_validation_data():
  """
  Delete validation data by moving images back from the validation directory
  to the data directory and deleting their subdirectories. 
  """
  validationsets = os.listdir(validation_dir)
  for v in validationsets:
    images = os.listdir(os.path.join(validation_dir, v))
    for i in images: 
      os.rename(os.path.join(validation_dir, v, i), os.path.join(data_dir, v,i))
    os.rmdir(os.path.join(validation_dir, v))

##############################################################################

remove_validation_data()
ndat, nval = create_validation_data(10)

# dimensions of our images. if your raw data is not of this size,
# you will need to use ffmpeg (or some other tool) to figure out
#how to reformat your data
img_width, img_height = 256, 256

# this is where your training and testing data (images) need to be relatve to
# where your script (altimeter_nn.py) is
train_data_dir = data_dir
validation_data_dir = validation_dir
nb_train_samples = ndat
nb_validation_samples = nval
epochs = 8
# Check to see if it shuffles batch_size randomly or sequentially
batch_size = 36

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Google "keras model sequential add" to understand what this code is doing
model = Sequential()
model.add(Conv2D(32, (6, 6), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(9, 9)))

model.add(Conv2D(64, (6, 6)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(94, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
                                   rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
                                                    train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
                                                        validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size)

print(model.summary())


plot_model(model)


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show(block=False)
plt.savefig('fig1.png')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show(block=False)
plt.savefig('fig2.png')

model.save_weights('first_try.h5')

remove_validation_data()
