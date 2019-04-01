import keras
from keras.models import Sequential
from keras.layers import  Dense, Activation, Dropout#, GlobalAveragePooling1D, MaxPool1D, Embedding, Dropout, Conv1D
from keras.layers import MaxPool2D

import numpy as np


def create_model(classes=1, n=3):

    #dimension of output is gonna b n*(x,y,w,h) + classes
    model = Sequential()
    model.add(Conv2D(921600,kernel_size=3, activation='relu', input_shape=(1280,720,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(460792,kernel_size=3, activation='relu', input_shape=(640,360,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(230388,kernel_size=3, activation='relu', input_shape=(320,180,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(n*4+classes+1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n*4+classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.save('sample_model.h5')
    return model


def train(model,num_batches,epochs,test_dict={}):

    model.fit(test_dict['data'], test_dict['labels']
                batch_size=num_batches, epochs=epochs,
                verbose=1, validation_data=())
    score = model.evaluate(x_test,y_test,verbose=0)
    return

def update(model):
    model.save('sample_model.h5')
    return
