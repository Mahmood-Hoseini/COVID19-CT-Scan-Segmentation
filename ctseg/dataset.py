from __future__ import division, print_function

import os, glob, tqdm
import numpy as np
from math import ceil
from keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')


from . import patient


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
    for patient_dir in tqdm.tqdm(patient_dirs) :
        p = patient.PatientData(patient_dir)
        if (len(p.lungs_all) == len(p.cts_all) and 
            len(p.infects_all) == len(p.cts_all)) :
            cts_all += p.cts_all
            lungs_mask_all += p.lungs_all
            infects_mask_all += p.infects_all
        else :
            print('patient {:02d} does not have mask files.',
                  'Did not included.'.format(p.patient_id))

    return (np.asarray(cts_all)/255, \
            np.asarray(lungs_mask_all), \
            np.asarray(infects_mask_all), \
            p.image_size)


class Iterator(object):
    def __init__(self, cts_all, masks_all, batch_size,
                 shuffle=True, rotation_range=15,
                 width_shift_range=0.2, height_shift_range=0.2,
                 shear_range=0.1, zoom_range=0.01) :

        self.cts_all = cts_all
        self.masks_all = masks_all
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
        augmented_masks = []
        for n in self.index[start:end]:
            ct = self.cts_all[n]
            mask = self.masks_all[n]

            _, _, channels = ct.shape

            # stack image + mask together to simultaneously augment
            stacked = np.concatenate((ct, mask), axis=2)

            # apply simple affine transforms first using Keras
            augmented = self.img_generator.random_transform(stacked)

            # split image and mask back apart
            image = augmented[:, :, :channels]
            augmented_cts.append(image)
            image = np.round(augmented[:, :, channels:])
            augmented_masks.append(image)

        self.i += self.batch_size
        if self.i >= len(self.cts_all):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        augmented_cts = np.asarray(augmented_cts)
        augmented_masks = np.asarray(augmented_masks)

        return augmented_cts, augmented_masks


def create_generators(data_dir, batch_size, img_data=[], validation_split=0.0, 
                      shuffle_train_val=True, shuffle=True, seed=None,
                      augment_training=True, augment_validation=True, 
                      augmentation_args={}):
    """
    Geneate batches of image data.
    inputs:
        - data_dir: directory that contains patientXX folders
        - batch_size: batch size, int
        - img_data: data of cts (channel 0) and lung or infection mask (channel 1). If
          this data doesn't exist, load it from data_dir above.
        - validation_split: percentage of training data to hold out for validation, float
        - shuffle: shuffle data, boolean
        - augment_training: augment training data, boolean
        - augment_validation: augment validation data, boolean
    outputs:
        - train_generator: image data generator for training data
        - val_generator: image data generator for validation data
    """
    if len(img_data) == 0 :
        cts_all, lungs_mask_all, infects_mask_all, _ = load_images(data_dir)

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

        # fuse masks together to pass to ImageDataGenerator
        masks_all = np.concatenate((lungs_mask_all, infects_mask_all), axis=-1)
    else :
        print("Preparing mask images...")
        if img_data.shape[-1] == 2 :
            cts_all, masks_all = np.split(img_data, 2, axis=-1) 
        elif img_data.shape[-1] == 3 :
            cts_all, lungs_mask_all, infects_mask_all = np.split(img_data, 3, axis=-1) 
            # fuse masks together to pass to ImageDataGenerator
            masks_all = np.concatenate((lungs_mask_all, infects_mask_all), axis=-1)


    # split out last %(validation_split) of images as validation set
    split_index = int((1-validation_split) * len(cts_all))

    if augment_training:
        train_generator = Iterator(cts_all[:split_index], masks_all[:split_index], 
                                    batch_size, shuffle=shuffle, **augmentation_args)
    else:
        img_generator = ImageDataGenerator()
        train_generator = img_generator.flow(cts_all[:split_index], masks_all[:split_index], 
                                                    batch_size=batch_size, shuffle=shuffle)

    train_steps_per_epoch = ceil(split_index / batch_size)

    if validation_split > 0.0:
        if augment_validation:
            val_generator = Iterator(cts_all[split_index:], masks_all[split_index:],
                                     batch_size, shuffle=shuffle, **augmentation_args)
        else:
            img_generator = ImageDataGenerator()
            val_generator = img_generator.flow(cts_all[split_index:], masks_all[split_index:], 
                                                    batch_size=batch_size, shuffle=shuffle)
    else:
        val_generator = None

    val_steps_per_epoch = ceil((len(cts_all) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch, val_generator, val_steps_per_epoch)
