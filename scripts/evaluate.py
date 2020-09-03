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
from scripts import train

def plot_and_save_ROC(model, cts, masks, figname) :
    mask_roc = []
    pred_masks = model.predict(cts)
    y_true = np.squeeze(masks.astype('int32')).ravel() # make it into vectors
    y_pred = np.squeeze(pred_masks.astype('float16')).ravel() # make it into vectors
    y_true = np.where(y_true>0.5, 1, 0) # binarizing true values
    fpr, tpr, _ = roc_curve(y_true, y_pred) 

    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, 'b', label='ROC (area=%0.2f)' %auc(fpr,tpr))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.savefig(figname, bbox_inches='tight')

    return np.reshape(fpr, [-1,1]), np.reshape(tpr, [-1,1])


def dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def jaccard(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[0,1,2])
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[0,1,2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)


def generate_stats(model, cts, masks):
    mask_dices = []
    mask_jaccard = []
    predictions = []
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    pred_masks = model.predict(cts)
    for ii in tqdm.tqdm(range(cts)) :
        y_true = masks[ii].astype('float16')
        y_pred = pred_masks[ii].astype('float16')
        mask_dices.append(dice(y_true, y_pred).eval(session=sess))
        mask_jaccard.append(jaccard(y_true, y_pred).eval(session=sess))

    print("Dice:    {:.3f} +/- {:.3f}".format(np.mean(mask_dices), np.std(mask_dices)))
    print("Jaccard: {:.3f} +/- {:.3f}".format(np.mean(mask_jaccard), np.std(mask_jaccard)))
    return np.reshape(mask_dices, [-1,1]), np.reshape(mask_jaccard, [-1,1])


def evaluate_now():
    args = opts.parse_arguments()

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
    # build lseg model to segment lungs
    string_to_model = { "convnet": models.lung_seg }
    model = string_to_model[args.model]
    lseg_model = model([height, width, channels], num_filters=args.lseg_filters, 
                            padding=args.padding)
    lseg_model.load_weights(os.path.join(args.outdir, args.lseg_outfile))

    # build model for infection segmentation
    string_to_model = { "convnet": models.infect_seg }
    model = string_to_model[args.model]
    iseg_model = model([height, width, channels], num_filters=args.iseg_filters, 
                    padding=args.padding, dropout=args.dropout)
    iseg_model.load_weights(os.path.join(args.outdir, args.iseg_outfile))


    print("Lung segmentation:")
    outfile = os.path.join(args.outdir, "lung-seg-ROC.pdf")
    lungs_fpr, lungs_tpr = plot_and_save_ROC(lseg_model, cts_all, lungs_mask_all, outfile)
    np.savetxt(args.outdir + "/lungs_roc.txt", np.concatenate((lungs_fpr, lungs_tpr), axis=1))
    lungs_dices, lungs_jacc = generate_stats(lseg_model, cts_all, lungs_mask_all)
    np.savetxt(args.outdir + "/lungs_stats.txt", np.concatenate((lungs_dices, lungs_jacc), axis=1))

    print("Infection segmentation:")
    pred_lungs = lseg_model.predict(cts_all, batch_size=args.batch_size)
    pred_lungs = tf.cast(pred_lungs+0.5, dtype=tf.int32)
    ccts_all, cinfects_all = train.prepare_img_data(cts_all, pred_lungs, 
                                                    infects_mask_all, img_size)
    outfile = os.path.join(args.outdir, "infection-seg-ROC.pdf")
    infects_fpr, infects_tpr = plot_and_save_ROC(iseg_model, ccts_all, cinfects_all, outfile)
    np.savetxt(args.outdir + "/infects_roc.txt", np.concatenate((infects_fpr, infects_tpr), axis=1))
    infects_dices, infects_jacc = generate_stats(iseg_model, ccts_all, cinfects_all)
    np.savetxt(args.outdir + "/infects_stats.txt", np.concatenate((infects_dices, infects_jacc), axis=1))


if __name__ == '__main__':
    evaluate_now()
