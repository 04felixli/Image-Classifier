import tensorflow as tf
import os
import cv2 
import imghdr # checks file extensions of images
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

# limit GPU memory use from tensorflow

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# filters out weirdly formatted images

data_dir = 'data' # variable to hold the path to data directory 

# os.listdir(data_dir) # returns folders inside of data directory (eg. "happy" and "sad")

os.listdir(os.path.join(data_dir, 'deadlift')) # returns every single image inside of happy folder
image_exts = ['jpeg', 'jpg', 'bmp', 'png'] # create list/array of image extension types

"""
img = (cv2.imread(os.path.join('data', 'happy', 'OIP.Qvl5obVH7DwngVCHoDxr2QHaHT.jpg'))) # reads an image from the happy folder inside of data folder
print(img.shape) # prints the size of the image 
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # paints the actual image and convert it to RGB color
plt.show() # print the actual image to the screen

"""

for image_class in os.listdir(data_dir): # loops through folders inside of data (eg. benchpress and deadlift)
    for image in os.listdir(os.path.join(data_dir, image_class)): # print every image in folders inside data
        image_path = os.path.join(data_dir, image_class, image) # set image_path to the location of every single image
        try:
            img = cv2.imread(image_path) # check if image can be loaded into opencv
            tip = imghdr.what(image_path) # check if image extension matches the ones in image_exts
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path) # deletes file
        except Exception as e:
            print('Issue with image{}'.format(image_path))

# load data/visualization?????

data = tf.keras.utils.image_dataset_from_directory('data') # builds image data set through data pipelines. stores them in variable called data.
# however, the data set is not automatically loaded into memory, all it does is generate the data. To allow access the data, we convert the data into a numpy iterator, below.
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next() # actually accesses the data pipeline. Grabs one batch of data. The batch is composed of 2 parts.
                             # Images are stored in batch[0], and labels are stored in batch[1]

# images represented as numpy arrays
print(batch[0].shape)

# prints a series of 1 and 0, corresponding to either happy or sad. But we do not know which is which...
# class 1 = deadlift
# class 0 = benchpress
print(batch[1]) 

# double check which class (deadlift or benchpress) is assigned to which number (0 or 1)
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()

# preprocess data

# scaling images
data = data.map(lambda x,y: (x/255, y)) # x is images, y is labels. What this does is:
                                        # as a batch is loaded, we get our images, and scale them by 255 to make them between 0 and 1
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

# split data into training and testing partition
train_size = int(len(data)*.7) # allocate 70% of batches to train 
val_size = int(len(data)*.2)+1 # allocate 20% of batches to validate
test_size = int(len(data)*.1)+1 # allocate 10% of batches to test

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# build deep learning model using keras sequential API

# build deep learning model
model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# train model

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# plot performance

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# evaluate performance

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# loop through each batch in testing data
for batch in test.as_numpy_iterator():
    X, y = batch # unpack the batch. x is the set of images, y is the y_true value???
    yhat = model.predict(X) # make the prediction
    pre.update_state(y, yhat) 
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

# test with random data

img = cv2.imread('benchpress_test.jpg')
resize = tf.image.resize(img, (256, 256))
# plt.imshowcv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(resize.numpy().astype(int))
plt.show()

np.expand_dims(resize, 0)
yhat = model.predict(np.expand_dims(resize/255, 0))

print (yhat)

if yhat > 0.5:
    print("Prediction: deadlift")
else:
    print("Prediction: benchpress")

"""
model.save(os.path.join('models', 'happysadmodel.h5'))
new_model = load_model(os.path.join('models', 'happysadmodel.h5'))
yhatnew = new_model.predict(np.expand_dims(resize/255, 0))

if yhatnew > 0.5:
    print("The image shows a sad person")
else:
    print("The image shows a happy person")
"""

