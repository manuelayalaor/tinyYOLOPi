# import tensorflow
# from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import  Dense, Activation, Dropout#, GlobalAveragePooling1D, MaxPool1D, Embedding, Dropout, Conv1D
from keras.layers import MaxPool2D

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

def prep_data():
    bboxes = np.zeros((num_imgs, num_objects, 4))
    imgs = np.zeros((num_imgs, img_size, img_size, 4), dtype=np.uint8)  # format: BGRA
    shapes = np.zeros((num_imgs, num_objects), dtype=int)
    return bboxes, imgs, shapes

def loss_func_params(_class_pred=[], class_labels=[]):#based on YOLO paper
    params = {}
    lam_coord = 5
    lam_no_obj = .5
    prob_class = 0.0
    pred_prob_class = 0.0
    return params

def loss_func(params={}):


    return ans

def train_model():
    bboxes, imgs, shapes = prep_data()
    num_imgs = 50000

    img_size = 32
    min_object_size = 4
    max_object_size = 16
    num_objects = 2
    num_shapes = 3
    shape_labels = ['rectangle', 'circle', 'triangle']
    for i_img in range(num_imgs):
        surface = cairo.ImageSurface.create_for_data(imgs[i_img], cairo.FORMAT_ARGB32, img_size, img_size)
        cr = cairo.Context(surface)

        # Fill background white.
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        # TODO: Try no overlap here.
        # Draw random shapes.
        for i_object in range(num_objects):
            shape = np.random.randint(num_shapes)
            shapes[i_img, i_object] = shape
            if shape == 0:  # rectangle
                w, h = np.random.randint(min_object_size, max_object_size, size=2)
                x = np.random.randint(0, img_size - w)
                y = np.random.randint(0, img_size - h)
                bboxes[i_img, i_object] = [x, y, w, h]
                cr.rectangle(x, y, w, h)
            elif shape == 1:  # circle
                r = 0.5 * np.random.randint(min_object_size, max_object_size)
                x = np.random.randint(r, img_size - r)
                y = np.random.randint(r, img_size - r)
                bboxes[i_img, i_object] = [x - r, y - r, 2 * r, 2 * r]
                cr.arc(x, y, r, 0, 2*np.pi)
            elif shape == 2:  # triangle
                w, h = np.random.randint(min_object_size, max_object_size, size=2)
                x = np.random.randint(0, img_size - w)
                y = np.random.randint(0, img_size - h)
                bboxes[i_img, i_object] = [x, y, w, h]
                cr.move_to(x, y)
                cr.line_to(x+w, y)
                cr.line_to(x+w, y+h)
                cr.line_to(x, y)
                cr.close_path()

            # TODO: Introduce some variation to the colors by adding a small random offset to the rgb values.
            color = np.random.randint(num_colors)
            colors[i_img, i_object] = color
            max_offset = 0.3
            r_offset, g_offset, b_offset = max_offset * 2. * (np.random.rand(3) - 0.5)
            if color == 0:
                cr.set_source_rgb(1-max_offset+r_offset, 0+g_offset, 0+b_offset)
            elif color == 1:
                cr.set_source_rgb(0+r_offset, 1-max_offset+g_offset, 0+b_offset)
            elif color == 2:
                cr.set_source_rgb(0+r_offset, 0-max_offset+g_offset, 1+b_offset)
            cr.fill()

        imgs = imgs[..., 2::-1]  # is BGRA, convert to RGB

        # surface.write_to_png('imgs/{}.png'.format(i_img))
        return imgs.shape, bboxes.shape, shapes.shape, colors.shape

if __name__ == "__main__":
    simple_model()
