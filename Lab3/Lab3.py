import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
import numpy as np

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

from PIL import Image
import os

import glob

folder = (glob.glob ("LAB3/*"))

files = []
for i in folder:
    files.append(glob.glob (i + "\\*"))

data_num = 0
for i in range(0,10):
    data_num = data_num +len(files[i])
    print(len(files[i]))
    
print("data_num = ", data_num)

x_train = []
y_train = []
x_test = []
y_test = []
x_origin = []
y_origin = []
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

crop_weight = 99
crop_height = 113


count = 0
for i in range(0,10):
    for myFile in files[i]:
        # print(count, myFile)
        count = count + 1
        im = Image.open(myFile)
        # lt.imshow(im)
        # lt.show()
        # print("A",type(im))
        im = im.convert("L") # 轉灰階
        # print("B",type(im))
        im = im.resize((crop_weight,crop_height), Image.ANTIALIAS)
        # print("B",type(im))
        x_origin.append(im)
        y_origin.append(i)
        x_train.append (im)
        y_train.append (i)

import random
divide_num = (int)(len(x_origin)/5)
print("take" , divide_num , "to be test sample")
rand_list = []
rand_list = random.sample(range(len(x_origin)), divide_num)
#print(random.sample(range(len(x_data)), (int)(len(x_data)/10)))
print('num of rand_list', len(rand_list))
print(rand_list)

rand_list.sort()

print(len(x_train))
print(len(y_train))

print(len(x_test))
print(len(y_test))

print(len(x_origin))
print(len(y_origin))

print(len(x_train))
print("mum" , "index", "y_data")

for i in range(0,len(rand_list)):
    print(i, rand_list[i], y_train[rand_list[i]])
    x_test.append(x_train[rand_list[i]])
    y_test.append(y_train[rand_list[i]])
    
for i in range(len(rand_list)-1, -1, -1):
    del_index = rand_list[i]
    print("del", del_index,"th")
    del x_train[del_index]
    del y_train[del_index]

print(len(x_train))
print(len(y_train))

print(len(x_test))
print(len(y_test))

print(len(x_origin))
print(len(y_origin))

nd3d_list = []
for i in range(0, len(x_origin)):
    nd3d_list.append(np.array(x_origin[i]))
x_origin_nd = np.array(nd3d_list)

print("shape of x_origin_nd : ", x_origin_nd.shape)

x_origin_nd = x_origin_nd.reshape(len(x_origin_nd), crop_height, crop_weight, 1)
print("shape of x_origin_nd : ", x_origin_nd.shape)

nd3d_list = []
for i in range(0, len(x_train)):
    nd3d_list.append(np.array(x_train[i]))
x_train_nd = np.array(nd3d_list)

print("shape of x_train_nd : ", x_train_nd.shape)

x_train_nd = x_train_nd.reshape(len(x_train_nd), crop_height, crop_weight, 1)
print("shape of x_train_nd : ", x_train_nd.shape)

nd3d_list = []
for i in range(0, len(x_test)):
    nd3d_list.append(np.array(x_test[i]))
x_test_nd = np.array(nd3d_list)

print("shape of x_test_nd : ", x_test_nd.shape)

x_test_nd = x_test_nd.reshape(len(x_test_nd), crop_height, crop_weight, 1)
print("shape of x_test_nd : ", x_test_nd.shape)

for i in range(0, 50):
    print(x_train_nd.shape)

    reshape_im = x_train_nd[i].reshape(crop_height, crop_weight)
    print(reshape_im.shape)

    plt.imshow(reshape_im, cmap='gray')
    # plt.show()

y_origin_2dlist = []
for i in range(0, len(y_origin)):
    y_origin_2dlist.append([y_origin[i]])

print("type of y_origin_2dlist : ", type(y_origin_2dlist))
print("type of y_origin_2dlist[0] : ", type(y_origin_2dlist[0]))
print("type of y_origin_2dlist[0][0] : ", type(y_origin_2dlist[0][0]))

y_train_2dlist = []
for i in range(0, len(y_train)):
    y_train_2dlist.append([y_train[i]])

print("type of y_train_2dlist : ", type(y_train_2dlist))
print("type of y_train_2dlist[0] : ", type(y_train_2dlist[0]))
print("type of y_train_2dlist[0][0] : ", type(y_train_2dlist[0][0]))

y_test_2dlist = []
for i in range(0, len(y_test)):
    y_test_2dlist.append([y_test[i]])

print("type of y_test_2dlist : ", type(y_test_2dlist))
print("type of y_test_2dlist[0] : ", type(y_test_2dlist[0]))
print("type of y_test_2dlist[0][0] : ", type(y_test_2dlist[0][0]))

y_test_nd = np.array(y_test_2dlist)
y_train_nd = np.array(y_train_2dlist)
y_origin_nd = np.array(y_origin_2dlist)
print("type of y_test_nd : \t", type(y_test_nd))
print("type of y_train_nd : \t", type(y_train_nd))
print("type of y_origin_nd : \t", type(y_origin_nd))
print("shape of y_test_nd : \t", y_test_nd.shape)
print("shape of y_train_nd : \t", y_train_nd.shape)
print("shape of y_origin_nd : \t", y_origin_nd.shape)

x_origin_nd = x_origin_nd.astype('float32')
x_origin_nd /= 255

x_train_nd = x_train_nd.astype('float32')
x_train_nd /= 255

x_test_nd = x_test_nd.astype('float32')
x_test_nd /= 255

batch_size = 64
num_classes = 10
epochs = 20
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'face_models')
model_name = 'face_model_weight_1.h5'
#SoFarBest_model_name = 'keras_kidney_SoFarBest_model_reduceLR_kernelRegularizer.h5'

# One-Hot Encoding
# Convert class vectors to binary class matrices.
y_origin_nd = keras.utils.to_categorical(y_origin_nd, num_classes)
y_train_nd = keras.utils.to_categorical(y_train_nd, num_classes)
y_test_nd = keras.utils.to_categorical(y_test_nd, num_classes)

print('y_origin_nd shape:', y_origin_nd.shape ,'\ny_origin_nd type:', type(y_origin_nd))
print('y_train_nd shape:', y_train_nd.shape ,'\ny_train_nd type:', type(y_train_nd))
print('y_test_nd shape:', y_test_nd.shape ,'\ny_test_nd type:', type(y_test_nd))

# dimensions of our images
#img_width, img_height = 332, 500

model = Sequential()  
model.add(Conv2D(96,(11,11),strides=(4,4),input_shape = x_train_nd.shape[1:],padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
#model.add(Dropout(0.25))

model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(num_classes,activation='softmax'))
model.summary()

#model.compile(loss='categorical_crossentropy',
#              optimizer='Adam',
#              metrics=['accuracy'])
model.compile(#loss='categorical_crossentropy',
              loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-5, nesterov=False),
              #optimizer=keras.optimizers.Adam(),
              #optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

'''
print('Not using data augmentation.')
model.fit(x_origin_nd, y_origin_nd,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          #validation_data=(x_test_nd, y_test_nd),
          shuffle=True,
          validation_split=0.2 )
'''


from keras.preprocessing.image import ImageDataGenerator

print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        fill_mode='nearest',  # set mode for filling points outside the input boundaries
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rescale=None,  # set rescaling factor (applied before any other transformation)
        preprocessing_function=None,  # set function that will be applied on each input
        data_format=None,  # image data format, either "channels_first" or "channels_last"
        validation_split=0.0)  # fraction of images reserved for validation (strictly between 0 and 1)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train_nd)
# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x_train_nd, y_train_nd,
                                 batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test_nd, y_test_nd),
                    workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test_nd, y_test_nd, verbose=1)
print('Test data loss:', scores[0])
print('Test data accuracy:', scores[1])

# Score trained model.
scores = model.evaluate(x_train_nd, y_train_nd, verbose=1)
print('Test data loss:', scores[0])
print('Test data accuracy:', scores[1])
