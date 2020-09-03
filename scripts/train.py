#!/usr/bin/env python

from __future__ import division, print_function

import os, glob
import argparse
import logging
import numpy as np
import tensorflow as tf
from keras import losses, optimizers, utils
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.optimizers import Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import cv2 as cv

import warnings
warnings.filterwarnings('ignore')


from ctseg import dataset, models, loss, opts, patient


def dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def jaccard(y_true, y_pred):
    batch_jaccard_coefs = loss.jaccard(y_true, y_pred, axis=[1, 2])
    jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
    return jaccard_coefs[1]


def select_optimizer(optimizer_name, optimizer_args):
    optimizers = {
        'adam': Adam,
        'adamax': Adamax,
        'sgd': SGD,
        'rmsprop': RMSprop,
        'nadam': Nadam,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
    }
    if optimizer_name not in optimizers:
        raise Exception("Unknown optimizer ({}).".format(optimizer_name))

    return optimizers[optimizer_name](**optimizer_args)


def select_metrics(metrics_name):
    metrics = {
        'accuracy': 'accuracy',
        'dice': dice,
        'jaccard': jaccard,
    }
    if metrics_name not in metrics:
        raise Exception("Unknown metrics ({}).".format(metrics_name))

    return metrics[metrics_name]


def prepare_img_data(cts, pred_lungs, infects, img_size) :
    ccts = []
    cinfects = []
    bad_ids = []
    for ii in range(len(cts)) :
        try :
            ct_img = np.asarray(cts[ii, :, :, 0])
            lung_img = pred_lungs[ii, :, :, 0].eval(session=tf.Session())
            infect_img = np.asarray(infects[ii, :, :, 0])

            bounds = patient.make_lungmask_bbox(ct_img, lung_img, display_flag=False)

            cct_img = patient.crop_(ct_img, bounds)
            img = cv.resize(cct_img, dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
            img = np.reshape(img, (img_size, img_size, 1))
            ccts.append(img)

            cinfect_img = patient.crop_(infect_img.astype('float32'), bounds)
            img = cv.resize(cinfect_img, dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
            img = np.reshape(img, (img_size, img_size, 1))
            cinfects.append(img)
        except :
            bad_ids.append(ii)
    # print(cts.shape, len(ccts), len(cinfects), len(bad_ids))
    return np.asarray(ccts), np.asarray(cinfects)


def train_now(args, train_for='lungs') :
    """
    Geneate batches of image data.
    inputs:
        - args: arguments prepared using opts.parse_arguments
        - train_for: string. either 'lungs' or 'infections'
    outputs:
        - None. Saves final weights in output directory.
    """
    if train_for not in ['lungs', 'infections'] :
        raise Exception("Unknown train_for. It must be 'lungs' or 'infections'.")

    print("Loading dataset...")
    cts_all, lungs_mask_all, infects_mask_all, img_size = dataset.load_images(args.datadir)

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.shuffle_train_val:
        # shuffle images and masks in parallel
        rng_state = np.random.get_state()
        np.random.shuffle(cts_all)
        np.random.set_state(rng_state)
        np.random.shuffle(lungs_mask_all)
        np.random.set_state(rng_state)
        np.random.shuffle(infects_mask_all)

    # get image dimensions 
    height, width, channels = cts_all[0].shape

    print("Building segmentation model...")
    if train_for == 'lungs' :
        string_to_model = { "convnet": models.lung_seg }
        model = string_to_model[args.model]
        model_ = model([height, width, channels], num_filters=args.lseg_filters, 
                        padding=args.padding)
    else :
        # build lseg model to segment lungs
        string_to_model = { "convnet": models.lung_seg }
        model = string_to_model[args.model]
        lseg_model = model([height, width, channels], num_filters=args.lseg_filters, 
                                padding=args.padding)
        lseg_model.load_weights(os.path.join(args.outdir, args.lseg_outfile))

        # build model for infection segmentation
        string_to_model = { "convnet": models.infect_seg }
        model = string_to_model[args.model]
        model_ = model([height, width, channels], num_filters=args.iseg_filters, 
                        padding=args.padding, dropout=args.dropout)

    model_.summary()

    if args.load_weights:
        logging.info("Loading saved weights from file: {}".format(args.load_weights))
        model_.load_weights(args.load_weights)

    # instantiate optimizer, and only keep args that have been set
    optimizer_args = {
        'lr':       args.learning_rate,
        'momentum': args.momentum,
        'decay':    args.decay
    }
    for k in list(optimizer_args):
        if optimizer_args[k] is None:
            del optimizer_args[k]
    optimizer = select_optimizer(args.optimizer, optimizer_args)

    ##### select loss function
    if args.loss == 'pixel':
        def lossfunc(y_true, y_pred):
            return loss.weighted_categorical_crossentropy(y_true, y_pred, args.loss_weights)
    elif args.loss == 'dice':
        def lossfunc(y_true, y_pred):
            return loss.sorensen_dice_loss(y_true, y_pred, args.loss_weights)
    elif args.loss == 'bce_dice':
        def lossfunc(y_true, y_pred):
            return loss.bce_dice_loss(y_true, y_pred, args.loss_weights)
    elif args.loss == 'jaccard':
        def lossfunc(y_true, y_pred):
            return loss.jaccard_loss(y_true, y_pred, args.loss_weights)
    else:
        raise Exception("Unknown loss ({})".format(args.loss))

    metrics = select_metrics(args.metrics)

    model_.compile(optimizer=optimizer, loss=lossfunc, metrics=[metrics])

    ##### automatic saving of model during training
    if args.checkpoint:
        if args.loss == 'pixel':
            filepath = os.path.join(args.outdir, 
                    "weights-{epoch:02d}-{val_acc:.4f}.hdf5")
            monitor = 'val_acc'
            mode = 'max'
        elif args.loss == 'bce_dice':
            filepath = os.path.join(args.outdir, 
                    "weights-{epoch:02d}-{val_dice:.4f}.hdf5")
            monitor='val_dice'
            mode = 'max'
        elif args.loss == 'jaccard':
            filepath = os.path.join(args.outdir, 
                    "weights-{epoch:02d}-{val_jaccard:.4f}.hdf5")
            monitor='val_jaccard'
            mode = 'max'

        checkpoint = ModelCheckpoint(filepath, monitor=monitor, 
                                     verbose=1, save_best_only=True, mode=mode)
        callbacks = [checkpoint]
    else:
        print("No model checkpoint set...")
        callbacks = []

    ### augmentation parameters
    augmentation_args = {
        'height_shift_range': args.height_shift_range,
        'width_shift_range': args.width_shift_range,
        'rotation_range': args.rotation_range,
        'zoom_range': args.zoom_range,
        'shear_range': args.shear_range,
    }

    # Beging train
    if train_for == 'lungs' :
        print("Preparing Image Data Generators...")
        img_data = np.concatenate((cts_all, lungs_mask_all), axis=-1)
        train_generator, train_steps_per_epoch, \
            val_generator, val_steps_per_epoch = dataset.create_generators(
                args.datadir, args.batch_size, img_data=img_data,
                validation_split=args.validation_split,
                shuffle_train_val=args.shuffle_train_val,
                shuffle=args.shuffle,
                seed=args.seed,
                augment_training=args.augment_training,
                augment_validation=args.augment_validation,
                augmentation_args=augmentation_args)
        outfname = args.lseg_outfile
    else :
        print("Cropping CTs for lung area...")
        pred_lungs = lseg_model.predict(cts_all, batch_size=args.batch_size)
        pred_lungs = tf.cast(pred_lungs+0.5, dtype=tf.int32)
        ccts_all, cinfects_all = prepare_img_data(cts_all, pred_lungs, 
                                                    infects_mask_all, img_size)

        print("Preparing Image Data Generators...")
        img_data = np.concatenate((ccts_all, cinfects_all), axis=-1)
        train_generator, train_steps_per_epoch, \
            val_generator, val_steps_per_epoch = dataset.create_generators(
                args.datadir, args.batch_size, img_data=img_data,
                validation_split=args.validation_split,
                shuffle_train_val=args.shuffle_train_val,
                shuffle=args.shuffle,
                seed=args.seed,
                augment_training=args.augment_training,
                augment_validation=args.augment_validation,
                augmentation_args=augmentation_args)
        outfname = args.iseg_outfile

    logging.info("Begin training...")
    model_.fit_generator(train_generator,
                        epochs=args.epochs,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_data=val_generator,
                        validation_steps=val_steps_per_epoch,
                        callbacks=callbacks,
                        verbose=1)

    model_.save(os.path.join(args.outdir, outfname))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = opts.parse_arguments()
    train_now(args, train_for='lungs')
