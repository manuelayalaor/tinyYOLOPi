import keras
from keras.models import Sequential
from keras.layers import  Dense, Activation, Dropout#, GlobalAveragePooling1D, MaxPool1D, Embedding, Dropout, Conv1D
from keras.layers import MaxPool2D

import cv2
import numpy as np

def create_model(classes=1, num_boxes=3):
    #dimension of output is gonna be num_boxes*(x,y,w,h) + classes
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
    model.add(Dense(num_boxes*4+classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.save('sample_model.h5')
    return model


def train(model, test_images,test_labels,epochs):
    model.fit(test_images, test_labels, epochs)
    test_loss , test_acc = model.evaluate(test_images, test_labels)
    #score = model.evaluate(x_test,y_test,verbose=0)
    return test_loss, test_acc

def test(model, test_images):
    return model.predict(test_images) #returns params of last layer

def update(model):
    model.save('sample_model.h5')
    return

def open_images(dir):
    img_path = os.path.relpath(dir)
    with os.scandir(img_path) as img_dir:
        yield cv2.imread(img_dir)


#change this to be a generator
def extract_labels_csv(file,FRAMES_TO_SAVE=30):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        bounding_boxes = [row for idx,row in enumerate(csv_reader) if idx % FRAMES_TO_SAVE == 0]
    return bounding_boxes #list of lists, convert to list of tuples


def main():
    #load the model
    model = keras.model.load("model\sample_model.h5")
    test_labels =  extract_labels_csv(r"preprocess\pedestrian-dataset\crosswalk.csv")
    test_images = open_images(r"preprocess\pedestrian-dataset\crosswalk-images") #FIX THIS

    #draw bounding boxes from labels here
    for img, labl in zip(test_images, test_labels):
        cv2.rectangle(img,(x,y),(w,h),(0,255,0),3)#fix this cause undefined
    train = (model,test_images,test_labels,epochs=10)
    print(f'Accuracy:{},loss:{}' test(model,test_images))

if __name__=='__main__':
    main()
