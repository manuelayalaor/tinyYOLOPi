# import tensorflow
# from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import  Dense, Activation, Dropout#, GlobalAveragePooling1D, MaxPool1D, Embedding, Dropout, Conv1D
from keras.layers import MaxPool2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import  Dense, GlobalAveragePooling1D, MaxPool1D, Embedding, Dropout, Conv1D
# from tensorflow.python.keras.layers import MaxPool2D

def simple_model():
    model = Sequential([
        Dense(200, input_dim=64),
        Activation('relu'),
        Dropout(0.2),
        Dense(4)
    ])
    model.compile('adadelta', 'mse')
    model.save('sample_model.h5')
    return model

def existing_model():
    model = load_model('sample_model.h5')
    return model

if __name__ == "__main__":
    simple_model()
