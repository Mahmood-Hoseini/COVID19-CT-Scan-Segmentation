#!/usr/bin/env python

from __future__ import division, print_function

import os, glob
import argparse
import logging
from keras import losses, optimizers, utils
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.optimizers import Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')


from ctseg import dataset, models, loss, opts



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


def train_now():
    logging.basicConfig(level=logging.INFO)

    args = opts.parse_arguments()

    logging.info("Loading dataset...")
    augmentation_args = {
        'height_shift_range': args.height_shift_range,
        'width_shift_range': args.width_shift_range,
        'rotation_range': args.rotation_range,
        'zoom_range': args.zoom_range,
        'shear_range': args.shear_range,
    }

    train_generator, train_steps_per_epoch, \
        val_generator, val_steps_per_epoch = dataset.create_generators(
            args.datadir, args.batch_size,
            validation_split=args.validation_split,
            shuffle_train_val=args.shuffle_train_val,
            shuffle=args.shuffle,
            seed=args.seed,
            augment_training=args.augment_training,
            augment_validation=args.augment_validation,
            augmentation_args=augmentation_args)

    # get image dimensions from the first batch
    cts, lungs_infects_mask_dict = next(train_generator)
    height, width, channels = cts[0].shape

    logging.info("Building model...")
    string_to_model = {
        "convnet": models.cts_model,
    }
    model = string_to_model[args.model]
    model_ = model([height, width, channels], num_filters=args.num_filters, 
                padding=args.padding, dropout=args.dropout)

    model_.summary()

    if args.load_weights:
        logging.info("Loading saved weights from file: {}".format(args.load_weights))
        model_.load_weights(args.load_weights)

    # instantiate optimizer, and only keep args that have been set
    # (not all optimizers have args like `momentum' or `decay')
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

    loss_weight_dict = {'lung_output': args.output_weights[0], 
                        'infect_output': args.output_weights[1]}

    loss_dict = {'lung_output': lossfunc, 'infect_output': lossfunc}

    model_.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weight_dict, metrics=[metrics])

    ##### automatic saving of model during training
    if args.checkpoint:
        if args.loss == 'pixel':
            filepath = os.path.join(args.outdir, 
                    "weights-{epoch:02d}-{val_infect_output_acc:.4f}.hdf5")
            monitor = 'val_infect_output_acc'
            mode = 'max'
        elif args.loss == 'bce_dice':
            filepath = os.path.join(args.outdir, 
                    "weights-{epoch:02d}-{val_infect_output_dice:.4f}.hdf5")
            monitor='val_infect_output_dice'
            mode = 'max'
        elif args.loss == 'jaccard':
            filepath = os.path.join(args.outdir, 
                    "weights-{epoch:02d}-{val_infect_output_jaccard:.4f}.hdf5")
            monitor='val_infect_output_jaccard'
            mode = 'max'

        checkpoint = ModelCheckpoint(filepath, monitor=monitor, verbose=1, save_best_only=True, mode=mode)
        callbacks = [checkpoint]
    else:
        print("No model checkpoint set...")
        callbacks = []

    # train
    logging.info("Begin training...")
    ress = model_.fit_generator(train_generator,
                    epochs=args.epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    callbacks=callbacks,
                    verbose=1)
    print(ress.history.keys())
    model_.save(os.path.join(args.outdir, args.outfile))

if __name__ == '__main__':
    train_now()
