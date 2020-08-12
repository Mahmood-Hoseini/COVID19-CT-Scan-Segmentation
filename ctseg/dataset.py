from __future__ import division, print_function

import os, glob
import numpy as np
from math import ceil
from keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')


from . import patient


def make_multi_output_flow(orig_flow, lung_tensor, infect_tensor):
    while True:
        (X, y_next_i) = next(orig_flow)
        y_next = {'lung_output': lung_tensor[y_next_i], 
                  'infect_output': infect_tensor[y_next_i]} 
        yield X, y_next


def load_images(data_dir):
    """Load all patient CT scan images, lung and infection masks from 
    train-assets directory. The directories and images are read in sorted order.

    Arguments:
      - data_dir: path to data directory

    Output:
      - tuples of (cts, lung_masks, infection_masks): all data are 4-d tensors of shape
        (num_examples, height, width, channels).
    """

    glob_search = os.path.join(data_dir, "patient*")
    patient_dirs = sorted(glob.glob(glob_search))
    if len(patient_dirs) == 0:
        raise Exception("No patient directors found in {}".format(data_dir))

    # load all images into memory
    cts_all = []
    lungs_mask_all = []
    infects_mask_all = []
    for patient_dir in patient_dirs:
        p = patient.PatientData(patient_dir)
        cts_all += p.cts_all
        lungs_mask_all += p.lungs_all
        infects_mask_all += p.infects_all

    return cts_all, lungs_mask_all, infects_mask_all


class Iterator(object):
    def __init__(self, 
                 cts_all, 
                 lungs_mask_all, 
                 infects_mask_all,
                 batch_size,
                 shuffle=True,
                 rotation_range=15,
                 width_shift_range=0.2,
                 height_shift_range=0.2,
                 shear_range=0.1,
                 zoom_range=0.01) :

        self.cts_all = cts_all
        self.lungs_mask_all = lungs_mask_all
        self.infects_mask_all = infects_mask_all
        self.batch_size = batch_size
        self.shuffle = shuffle
        augment_options = {
            'rotation_range': rotation_range,
            'width_shift_range': width_shift_range,
            'height_shift_range': height_shift_range,
            'shear_range': shear_range,
            'zoom_range': zoom_range,
        }
        self.img_generator = ImageDataGenerator(**augment_options)
        self.i = 0
        self.index = np.arange(len(cts_all))
        if shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        return self.next()

    def next(self):
        # compute how many images to output in this batch
        start = self.i
        end = min(start+self.batch_size, len(self.cts_all))
        augmented_cts = []
        augmented_lungs_mask = []
        augmented_infects_mask = []
        for n in self.index[start:end]:
            ct = self.cts_all[n]
            lung_mask = self.lungs_mask_all[n]
            infect_mask = self.infects_mask_all[n]

            _, _, channels = ct.shape

            # stack image + mask together to simultaneously augment
            stacked = np.concatenate((ct, lung_mask, infect_mask), axis=2)

            # apply simple affine transforms first using Keras
            augmented = self.img_generator.random_transform(stacked)

            # split image and mask back apart
            image = augmented[:, :, :channels]
            augmented_cts.append(image)
            image = np.round(augmented[:, :, channels:2*channels])
            augmented_lungs_mask.append(image)
            image = np.round(augmented[:, :, 2*channels:])
            augmented_infects_mask.append(image)

        self.i += self.batch_size
        if self.i >= len(self.cts_all):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        augmented_output = {'lung_output': np.asarray(augmented_lungs_mask), 
                            'infect_output': np.asarray(augmented_infects_mask)}
        return np.asarray(augmented_cts), augmented_output


def create_generators(data_dir, batch_size, validation_split=0.0, 
                      shuffle_train_val=True, shuffle=True, seed=None,
                      augment_training=True, augment_validation=True, 
                      augmentation_args={}):
    """
    Geneate batches of image data.
    inputs:
        - data_dir: directory that contains patientXX folders
        - batch_size: batch size, int
        - validation_split: percentage of training data to hold out for validation, float
        - shuffle: shuffle data, boolean
        - augment_training: augment training data, boolean
        - augment_validation: augment validation data, boolean
    outputs:
        - train_generator: image data generator for training data
        - val_generator: image data generator for validation data
    """
    cts_all, lungs_mask_all, infects_mask_all = load_images(data_dir)

    if seed is not None:
        np.random.seed(seed)

    if shuffle_train_val:
        # shuffle images and masks in parallel
        rng_state = np.random.get_state()
        np.random.shuffle(cts_all)
        np.random.set_state(rng_state)
        np.random.shuffle(lungs_mask_all)
        np.random.set_state(rng_state)
        np.random.shuffle(infects_mask_all)

    # split out last %(validation_split) of images as validation set
    split_index = int((1-validation_split) * len(cts_all))

    if augment_training:
        train_generator = Iterator(cts_all[:split_index], lungs_mask_all[:split_index], 
                                    infects_mask_all[:split_index], batch_size, 
                                    shuffle=shuffle, **augmentation_args)
    else:
        img_generator = ImageDataGenerator()
        train_generator = img_generator.flow(cts_all[:split_index], range(split_index), 
                                                    batch_size=batch_size, shuffle=shuffle)
        train_generator = make_multi_output_flow(train_generator[:split_index], 
                                                 lungs_mask_all[:split_index], 
                                                 infects_mask_all[:split_index])

    train_steps_per_epoch = ceil(split_index / batch_size)

    if validation_split > 0.0:
        if augment_validation:
            val_generator = Iterator(cts_all[split_index:], lungs_mask_all[split_index:],
                                     infects_mask_all[split_index:], batch_size, 
                                     shuffle=shuffle, **augmentation_args)
        else:
            img_generator = ImageDataGenerator()
            val_generator = img_generator.flow(cts_all[split_index:], range(len(cts_all)-split_index), 
                                                    batch_size=batch_size, shuffle=shuffle)
            train_generator = make_multi_output_flow(train_generator[split_index:], 
                                                     lungs_mask_all[split_index:], 
                                                     infects_mask_all[split_index:])
    else:
        val_generator = None

    val_steps_per_epoch = ceil((len(cts_all) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch, val_generator, val_steps_per_epoch)
