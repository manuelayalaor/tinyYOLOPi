import tensorflow as tf
from tf import keras
from tf.keras.models import Sequential
from tf.keras.layers import  Dense, GlobalAveragePooling1D, MaxPool1D, Embedding, Dropout, Conv1D
from tf.python.keras.layers import MaxPool2D

global classes = 10
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                   metrics=['accuracy'])
    return model