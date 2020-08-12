
from __future__ import division, print_function

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')


from ctseg import opts, patient, dataset, models


def load_patient_images(path):
    p = patient.PatientData(path)
    cts = np.asarray(p.cts_all, dtype='float64')
    lungs = np.asarray(p.lungs_all, dtype='float64')
    infects = np.asarray(p.infects_all, dtype='float64')
    return cts, lungs, infects, p.patient_id


def compare_actual_and_predicted(X, yl_mask, yi_mask, yl_pred, yi_pred, 
                                 img_num, num_pix, figname) :
    fig = plt.figure(figsize=(12,7))
    plt.subplot(2,3,1)
    plt.imshow(np.reshape(X[img_num], [num_pix, num_pix]))
    plt.title('CT image')

    plt.subplot(2,3,2)
    plt.imshow(np.reshape(yl_mask[img_num], [num_pix, num_pix]), cmap='bone')
    plt.title('lung mask')
    plt.subplot(2,3,3)
    plt.imshow(np.reshape(yi_mask[img_num], [num_pix, num_pix]), cmap='bone')
    plt.title('infection mask')

    plt.subplot(2,3,5)
    plt.imshow(np.reshape(yl_pred[img_num], [num_pix, num_pix]), cmap='bone')
    plt.title('predicted lung mask')
    plt.subplot(2,3,6)
    plt.imshow(np.reshape(yi_pred[img_num], [num_pix, num_pix]), cmap='bone')
    plt.title('predicted infection mask')

    plt.savefig(figname, bbox_inches='tight')


def test_now():
    args = opts.parse_arguments()

    glob_search = os.path.join(args.testdir, "patient*")
    patient_dirs = sorted(glob.glob(glob_search))
    if len(patient_dirs) == 0:
        raise Exception("No patient directors found in {}".format(patient_dirs))

    # get image dimensions from first patient
    cts, lungs, infects, _ = load_patient_images(patient_dirs[0])
    height, width, channels = cts[0].shape

    print("Building model...")
    string_to_model = {
        "convnet": models.cts_model,
    }
    model = string_to_model[args.model]
    m = model([height, width, channels], num_filters=args.num_filters, 
                padding=args.padding, dropout=args.dropout)

    m.load_weights(os.path.join(args.outdir, args.outfile))

    for path in patient_dirs:
        cts, lungs, infects, patient_id = load_patient_images(path)
        num_slices = len(cts)

        yl_pred, yi_pred = m.predict(cts) 

        img_nums = random.choices(range(num_slices), k=min(4, num_slices))
        for img in img_nums :
            figname = 'actualvs.pred-patient{:02d}-frame{:03d}.pdf'.format(patient_id, img)
            outfile = os.path.join(args.outdir, figname)

            compare_actual_and_predicted(cts, lungs, infects, yl_pred, yi_pred, 
                                         img, height, outfile)

if __name__ == '__main__':
    test_now()
