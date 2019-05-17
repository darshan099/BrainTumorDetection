from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout,Flatten,Dense,Activation
from keras.layers.convolutional import Convolution2D,MaxPooling2D

#data paths
train_data_path="./data/training_data/"
test_data_path="./data/test_data"

#parameters
img_width,img_height=32,32
batch_size=32
samples_per_epoch=700
validation_steps=200
nb_filters1=32
nb_filters2=64
conv1_size=3
conv2_size=2
epochs=25
pool_size=2
classes_num=3
lr=0.0004



#initialize cnn
classifier=Sequential()

#convolution
classifier.add(Convolution2D(nb_filters1,conv1_size,conv1_size,border_mode="same",input_shape=(img_width,img_height,3)))
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size=(pool_size,pool_size)))

#adding second convolution layer
classifier.add(Convolution2D(nb_filters2,conv2_size,conv2_size,border_mode="same"))
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size=(pool_size,pool_size),dim_ordering='th'))

#flattening
classifier.add(Flatten())

#full connection
classifier.add(Dense(128))
classifier.add(Activation("relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(classes_num,activation='softmax'))

#compile cnn
classifier.compile(optimizer=optimizers.RMSprop(lr=lr),loss='categorical_crossentropy', metrics=['accuracy'])

#create data generator
train_datagen=ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen=ImageDataGenerator(
    rescale=1. / 255
)
train_generator=train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator=test_datagen.flow_from_directory(
    test_data_path,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

#fit method
classifier.fit_generator(
    train_generator,
    steps_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=validation_steps
)

classifier.save('./models/model.h5')
classifier.save_weights('./models/weights.h5')
