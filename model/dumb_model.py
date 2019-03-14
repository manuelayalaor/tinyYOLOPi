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

# Flip bboxes during training.
# Note: The validation loss is always quite big here because we don't flip the bounding boxes for the validation data.
def intersectionOverUnion(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]  # TODO: Check if its more performant if tensor elements are accessed directly below.
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0
    intersection = w_I * h_I

    union = w1 * h1 + w2 * h2 - I

    return intersection / union

def dist(bbox1, bbox2):
    return np.sqrt(np.sum(np.square(bbox1[:2] - bbox2[:2])))

def init_arrays(size, some_num):
    return [np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping)) for i in range(6)]

def predict_y(num_objects, num_epochs_flipping, train_X, flipped_train_y, test_X,test_y):
    for epoch in range(num_epochs_flipping):
        print 'Epoch', epoch
        model.fit(train_X, flipped_train_y, nb_epoch=1, validation_data=(test_X, test_y), verbose=2)
        pred_y = model.predict(train_X)

        for sample, (pred, exp) in enumerate(zip(pred_y, flipped_train_y)):

            # TODO: Make this simpler.
            pred = pred.reshape(num_objects, -1)
            exp = exp.reshape(num_objects, -1)

            pred_bboxes = pred[:, :4]
            exp_bboxes = exp[:, :4]

            ious = np.zeros((num_objects, num_objects))
            dists = np.zeros((num_objects, num_objects))
            mses = np.zeros((num_objects, num_objects))
            for i, exp_bbox in enumerate(exp_bboxes):
                for j, pred_bbox in enumerate(pred_bboxes):
                    ious[i, j] = intersectionOverUnion(exp_bbox, pred_bbox)
                    dists[i, j] = dist(exp_bbox, pred_bbox)
                    mses[i, j] = np.mean(np.square(exp_bbox - pred_bbox))

            new_order = np.zeros(num_objects, dtype=int)

            for i in range(num_objects):
                # Find pred and exp bbox with maximum iou and assign them to each other (i.e. switch the positions of the exp bboxes in y).
                ind_exp_bbox, ind_pred_bbox = np.unravel_index(ious.argmax(), ious.shape)
                ious_epoch[sample, epoch] += ious[ind_exp_bbox, ind_pred_bbox]
                dists_epoch[sample, epoch] += dists[ind_exp_bbox, ind_pred_bbox]
                mses_epoch[sample, epoch] += mses[ind_exp_bbox, ind_pred_bbox]
                ious[ind_exp_bbox] = -1  # set iou of assigned bboxes to -1, so they don't get assigned again
                ious[:, ind_pred_bbox] = -1
                new_order[ind_pred_bbox] = ind_exp_bbox

            flipped_train_y[sample] = exp[new_order].flatten()

            flipped[sample, epoch] = 1. - np.mean(new_order == np.arange(num_objects, dtype=int))#np.array_equal(new_order, np.arange(num_objects, dtype=int))  # TODO: Change this to reflect the number of flips.
            ious_epoch[sample, epoch] /= num_objects
            dists_epoch[sample, epoch] /= num_objects
            mses_epoch[sample, epoch] /= num_objects

            acc_shapes_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4:4+num_shapes], axis=-1) == np.argmax(exp[:, 4:4+num_shapes], axis=-1))
            acc_colors_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1) == np.argmax(exp[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1))


        # Calculate metrics on test data.
        return model.predict(test_X)

def train():
    num_objects = 2
    num_epochs_flipping = 50
    num_epochs_no_flipping = 0  # has no significant effect

    flipped_train_y = np.array(train_y)
    flipped = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    ious_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    dists_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    mses_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    acc_shapes_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    acc_colors_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    # all_train = [np.zeros(
    # (len(test_y), num_epochs_flipping + num_epochs_no_flipping)) for i in range(6)]

    flipped_test_y = np.array(test_y)
    flipped_test = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    ious_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    dists_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    mses_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    acc_shapes_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    acc_colors_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    #all_tests = [np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping)) for i in range(6)]

    # TODO: Calculate ious directly for all samples (using slices of the array pred_y for x, y, w, h).
    pred_test_y = pred_y(num_objects, num_epochs_flipping, train_X, flipped_train_y, test_X,test_y)
        # TODO: Make this simpler.
        for sample, (pred, exp) in enumerate(zip(pred_test_y, flipped_test_y)):

            # TODO: Make this simpler.
            pred = pred.reshape(num_objects, -1)
            exp = exp.reshape(num_objects, -1)

            pred_bboxes = pred[:, :4]
            exp_bboxes = exp[:, :4]

            ious = np.zeros((num_objects, num_objects))
            dists = np.zeros((num_objects, num_objects))
            mses = np.zeros((num_objects, num_objects))
            for i, exp_bbox in enumerate(exp_bboxes):
                for j, pred_bbox in enumerate(pred_bboxes):
                    ious[i, j] = IOU(exp_bbox, pred_bbox)
                    dists[i, j] = dist(exp_bbox, pred_bbox)
                    mses[i, j] = np.mean(np.square(exp_bbox - pred_bbox))

            new_order = np.zeros(num_objects, dtype=int)

            for i in range(num_objects):
                # Find pred and exp bbox with maximum iou and assign them to each other (i.e. switch the positions of the exp bboxes in y).
                ind_exp_bbox, ind_pred_bbox = np.unravel_index(mses.argmin(), mses.shape)
                ious_test_epoch[sample, epoch] += ious[ind_exp_bbox, ind_pred_bbox]
                dists_test_epoch[sample, epoch] += dists[ind_exp_bbox, ind_pred_bbox]
                mses_test_epoch[sample, epoch] += mses[ind_exp_bbox, ind_pred_bbox]
                mses[ind_exp_bbox] = 1000000#-1  # set iou of assigned bboxes to -1, so they don't get assigned again
                mses[:, ind_pred_bbox] = 10000000#-1
                new_order[ind_pred_bbox] = ind_exp_bbox

            flipped_test_y[sample] = exp[new_order].flatten()

            flipped_test[sample, epoch] = 1. - np.mean(new_order == np.arange(num_objects, dtype=int))#np.array_equal(new_order, np.arange(num_objects, dtype=int))  # TODO: Change this to reflect the number of flips.
            ious_test_epoch[sample, epoch] /= num_objects
            dists_test_epoch[sample, epoch] /= num_objects
            mses_test_epoch[sample, epoch] /= num_objects

            acc_shapes_test_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4:4+num_shapes], axis=-1) == np.argmax(exp[:, 4:4+num_shapes], axis=-1))
            acc_colors_test_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1) == np.argmax(exp[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1))


        print( 'Flipped {} % of all elements'.format(np.mean(flipped[:, epoch]) * 100.))
        print( 'Mean IOU: {}'.format(np.mean(ious_epoch[:, epoch])))
        print( 'Mean dist: {}'.format(np.mean(dists_epoch[:, epoch])))
        print( 'Mean mse: {}'.format(np.mean(mses_epoch[:, epoch])))
        print( 'Accuracy shapes: {}'.format(np.mean(acc_shapes_epoch[:, epoch])))
        print( 'Accuracy colors: {}'.format(np.mean(acc_colors_epoch[:, epoch])))

        print ('--------------- TEST ----------------')
        print ('Flipped {} % of all elements'.format(np.mean(flipped_test[:, epoch]) * 100.))
        print( 'Mean IOU: {}'.format(np.mean(ious_test_epoch[:, epoch])))
        print( 'Mean dist: {}'.format(np.mean(dists_test_epoch[:, epoch])))
        print( 'Mean mse: {}'.format(np.mean(mses_test_epoch[:, epoch])))
        print( 'Accuracy shapes: {}'.format(np.mean(acc_shapes_test_epoch[:, epoch])))
        print( 'Accuracy colors: {}'.format(np.mean(acc_colors_test_epoch[:, epoch])))
        print('')
        return

if __name__ == "__main__":
    simple_model()
