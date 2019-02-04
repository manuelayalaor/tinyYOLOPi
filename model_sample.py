import tensorflow as tf
from tf import keras
from tf.keras.models import Sequential
from tf.keras.layers import  Dense, GlobalAveragePooling1D, MaxPool1D, Embedding, Dropout, Conv1D
from tf.python.keras.layers import MaxPool2D

classes = 2 #HUMAN or not HUMAN
RANK = 3
DIM = 3

def create_model(input_shape=(1920,1080,1)):
    model = Sequential()
    model.add(Conv2D(units=32, kernel_size=(DIM,RANK),
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=classes, activation='softmax'))

    #configures learning process
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='rmsprop' ,#keras.optimizers.Adadelta(),
                   metrics=['accuracy'])
    return model
