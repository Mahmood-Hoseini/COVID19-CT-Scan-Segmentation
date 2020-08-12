from __future__ import division, print_function

import os, tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import roc_curve, auc 

import warnings
warnings.filterwarnings('ignore')


from ctseg import opts, patient, dataset, models


def plot_and_save_ROC(model, generator, steps_per_epoch) :
    lung_infect_roc = []
    counter = 0
    for ii in range(steps_per_epoch):
        cts, lungs_infects_mask = next(generator)
        lungs_mask = lungs_infects_mask['lung_output']
        infects_mask = lungs_infects_mask['infect_output']
        lung_mask_pred, infect_mask_pred = model.predict(cts)
        for yl_true, yi_true, yl_pred, yi_pred in zip(
            lungs_mask, infects_mask, lung_mask_pred, infect_mask_pred) :

            yl_true = np.squeeze(yl_true.astype('float16')).ravel() # make it into vectors
            yl_pred = np.squeeze(np.round(yl_pred).astype('float16')).ravel() # make it into vectors
            fprl, tprl, _ = roc_curve(yl_true, yl_pred)            

            yi_true = np.squeeze(yi_true.astype('float16')).ravel() # make it into vectors
            yi_pred = np.squeeze(np.round(yi_pred).astype('float16')).ravel() # make it into vectors
            fpri, tpri, _ = roc_curve(yi_true, yi_pred)
            lung_infect_roc = lung_infect_roc + fprl + tprl + fpri + tpri
            counter += 1

    fig, ax = plt.subplots(1,1)
    ax.plot(fprl, tprl, 'b', label='Lung ROC (area=%0.2f)' %auc(fprl,tprl))
    ax.plot(fpri, tpri, 'r', label='Infect ROC (area=%0.2f)' %auc(fpri,tpri))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show

    return lung_infect_roc, counter


def compare_actual_and_predicted(X, yl_mask, yi_mask, yl_pred, yi_pred, 
                                 num_pix, figname) :
    fig = plt.figure(figsize=(12,7))
    plt.subplot(2,3,1)
    plt.imshow(np.reshape(X, [num_pix, num_pix]))
    plt.title('CT image')

    plt.subplot(2,3,2)
    plt.imshow(np.reshape(yl_mask, [num_pix, num_pix]), cmap='bone')
    plt.title('lung mask')
    plt.subplot(2,3,3)
    plt.imshow(np.reshape(yi_mask, [num_pix, num_pix]), cmap='bone')
    plt.title('infection mask')

    plt.subplot(2,3,5)
    plt.imshow(np.reshape(yl_pred, [num_pix, num_pix]), cmap='bone')
    plt.title('predicted lung mask')
    plt.subplot(2,3,6)
    plt.imshow(np.reshape(yi_pred, [num_pix, num_pix]), cmap='bone')
    plt.title('predicted infection mask')

    plt.savefig(figname, bbox_inches='tight')


def dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def jaccard(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def generate_stats(model, generator, steps_per_epoch, return_images=False):
    lung_dices = []
    lung_jaccard = []
    infect_dices = []
    infect_jaccard = []
    predictions = []
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for ii in tqdm.tqdm(range(steps_per_epoch)):
        cts, lungs_infects_mask = next(generator)
        lungs_mask = lungs_infects_mask['lung_output']
        infects_mask = lungs_infects_mask['infect_output']
        lung_mask_pred, infect_mask_pred = model.predict(cts)
        for yl_true, yi_true, yl_pred, yi_pred in zip(
            lungs_mask, infects_mask, lung_mask_pred, infect_mask_pred) :

            yl_true = yl_true.astype('float16')
            yl_pred = np.round(yl_pred).astype('float16')
            lung_dices.append(dice(yl_true, yl_pred, smooth=1).eval(session=sess))
            lung_jaccard.append(jaccard(yl_true, yl_pred).eval(session=sess))

            yi_true = yi_true.astype('float16')
            yi_pred = np.round(yi_pred).astype('float16')
            infect_dices.append(dice(yi_true, yi_pred, smooth=1).eval(session=sess))
            infect_jaccard.append(jaccard(yi_true, yi_pred).eval(session=sess))

        if return_images:
            for ct, yl_true, yi_true, yl_pred, yi_pred in zip(cts, lungs_mask, 
                                        infects_mask, lung_mask_pred, infect_mask_pred):
                predictions.append((ct[:,:,0], yl_true[:,:,0], yi_true[:,:,0], 
                                               yl_pred[:,:,0], yi_pred[:,:,0]))
    print("Lung Dice:    {:.3f} +/- {:.3f}".format(np.mean(lung_dices), np.std(lung_dices)))
    print("Lung Jaccard:    {:.3f} +/- {:.3f}".format(np.mean(lung_jaccard), np.std(lung_jaccard)))
    print("Infection Dice:    {:.3f} +/- {:.3f}".format(np.mean(infect_dices), np.std(infect_dices)))
    print("Infection Jaccard:    {:.3f} +/- {:.3f}".format(np.mean(infect_jaccard), np.std(infect_jaccard)))
    return lung_dices, lung_jaccard, infect_dices, infect_jaccard, predictions


def evaluate_now():
    args = opts.parse_arguments()

    augmentation_args = {
        'rotation_range': args.rotation_range,
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'shear_range': args.shear_range,
        'zoom_range': args.zoom_range,
    }

    print("Loading dataset...")
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

    # get image dimensions from first batch
    cts, lungs_infects_mask = next(train_generator)
    height, width, channels = cts[0].shape

    print("Building model...")
    string_to_model = {
        "convnet": models.cts_model,
    }
    model = string_to_model[args.model]
    model_ = model([height, width, channels], num_filters=args.num_filters, 
                padding=args.padding, dropout=args.dropout)

    model_.load_weights(os.path.join(args.outdir, args.outfile))

    print("Training Set:")
    train_lung_infect_roc, counter = plot_and_save_ROC(model_, train_generator, train_steps_per_epoch)
    train_lung_infect_roc = np.reshape(np.asarray(train_lung_infect_roc), [-1,counter], order='F')
    np.savetxt(args.outdir + "/train_roc.txt", train_lung_infect_roc)
    train_lung_dices, train_lung_jacc, train_infect_dices, train_infect_jacc, \
            train_images = generate_stats(model_, train_generator, train_steps_per_epoch,
                            return_images=args.checkpoint)

    print("Validation Set:")
    val_lung_infect_roc, counter = plot_and_save_ROC(model_, val_generator, val_steps_per_epoch)
    val_lung_infect_roc = np.reshape(np.asarray(val_lung_infect_roc), [-1,counter], order='F')
    np.savetxt(args.outdir + "/val_roc.txt", val_lung_infect_roc)
    val_lung_dices, val_lung_jacc, val_infect_dices, val_infect_jacc, \
            val_images = generate_stats(model_, val_generator, val_steps_per_epoch,
                            return_images=args.checkpoint)

    if args.outfile:
        train_data = np.reshape(np.asarray(train_lung_dices + train_lung_jacc + 
                                           train_infect_dices, train_infect_jacc), [-1,4], order='F')
        val_data = np.reshape(np.asarray(val_lung_dices + val_lung_jacc + 
                                         val_infect_dices + val_infect_jacc), [-1,4], order='F')
        np.savetxt(args.outdir + "/train_stats.txt", train_data)
        np.savetxt(args.outdir + "/val_stats.txt", val_data)

    if args.checkpoint:
        print("Saving images...")
        for ii, dice in enumerate(train_infect_dices) :
            ct, yl_true, yi_true, yl_pred, yi_pred = train_images[ii]
            figname = "train-{:03d}-{:.3f}.png".format(ii, dice)
            figname = os.path.join(args.outdir, figname)
            compare_actual_and_predicted(ct, yl_true, yi_true, yl_pred, yi_pred, ct.shape[1], figname)
        for ii, dice in enumerate(val_infect_dices):
            ct, yl_true, yi_true, yl_pred, yi_pred = val_images[ii]
            figname = "val-{:03d}-{:.3f}.png".format(ii, dice)
            figname = os.path.join(args.outdir, figname)
            compare_actual_and_predicted(ct, yl_true, yi_true, yl_pred, yi_pred, ct.shape[1], figname)

if __name__ == '__main__':
    evaluate_now()
