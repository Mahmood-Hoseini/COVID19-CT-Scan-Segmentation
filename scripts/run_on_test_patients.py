
from __future__ import division, print_function

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')


from ctseg import opts, patient, dataset, models


def load_patient_images(path):
    p = patient.PatientData(path)
    cts = np.asarray(p.cts_all, dtype='float64')/255
    lungs = np.asarray(p.lungs_all, dtype='float64')
    infects = np.asarray(p.infects_all, dtype='float64')
    return cts, lungs, infects, p.patient_id, p.image_size


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


def prepare_img_data(cts, lungs, pred_lungs, infects, img_size) :
    ccts = []
    clungs = []
    cpred_lungs = []
    cinfects = []
    bad_ids = []
    for ii in range(len(cts)) :
        try :
            ct_img = np.asarray(cts[ii, :, :, 0])
            lung_img = np.asarray(lungs[ii, :, :, 0])
            plung_img = pred_lungs[ii, :, :, 0].eval(session=tf.Session())
            infect_img = np.asarray(infects[ii, :, :, 0])
            
            bounds = patient.make_lungmask_bbox(ct_img, plung_img, display_flag=False)

            cct_img = patient.crop_(ct_img, bounds)
            img = cv.resize(cct_img, dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
            img = np.reshape(img, (img_size, img_size, 1))
            ccts.append(img)

            clung_img = patient.crop_(lung_img.astype('float32'), bounds)
            img = cv.resize(clung_img, dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
            img = np.reshape(img, (img_size, img_size, 1))
            clungs.append(img)

            cplung_img = patient.crop_(plung_img.astype('float32'), bounds)
            img = cv.resize(cplung_img, dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
            img = np.reshape(img, (img_size, img_size, 1))
            cpred_lungs.append(img)

            cinfect_img = patient.crop_(infect_img.astype('float32'), bounds)
            img = cv.resize(cinfect_img, dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
            img = np.reshape(img, (img_size, img_size, 1))
            cinfects.append(img)
        except :
            bad_ids.append(ii)
    # print(cts.shape, len(ccts), len(clungs), len(cpred_lungs), len(cinfects), len(bad_ids))
    return np.asarray(ccts), np.asarray(clungs), np.asarray(cpred_lungs), np.asarray(cinfects)


def test_now():
    args = opts.parse_arguments()

    glob_search = os.path.join(args.testdir, "patient*")
    patient_dirs = sorted(glob.glob(glob_search))
    if len(patient_dirs) == 0:
        raise Exception("No patient directors found in {}".format(patient_dirs))

    # get image dimensions from first patient
    cts, lungs, infects, _, _ = load_patient_images(patient_dirs[0])
    height, width, channels = cts[0].shape

    print("Building models...")
    string_to_model = {"convnet": models.lung_seg}
    model = string_to_model[args.model]
    lseg_model = model([height, width, channels], num_filters=args.lseg_filters, 
                        padding=args.padding)
    lseg_model.load_weights(os.path.join(args.outdir, args.lseg_outfile))
    
    string_to_model = {"convnet": models.infect_seg}
    model = string_to_model[args.model]
    iseg_model = model([height, width, channels], num_filters=args.iseg_filters, 
                        padding=args.padding, dropout=args.dropout)
    iseg_model.load_weights(os.path.join(args.outdir, args.iseg_outfile))

    for path in patient_dirs:
        cts, lungs, infects, patient_id, img_size = load_patient_images(path)
        num_slices = len(cts)

        print("Cropping CTs for lung area...")
        pred_lungs = lseg_model.predict(cts, batch_size=args.batch_size)
        pred_lungs = tf.cast(pred_lungs+0.5, dtype=tf.int32)
        ccts, clungs, cpred_lungs, cinfects = prepare_img_data(cts, lungs, pred_lungs, 
                                                    infects, img_size)

        print("Predicting infections...")
        pred_infects = iseg_model.predict(ccts, batch_size=args.batch_size)
        pred_infects = tf.cast(pred_infects+0.5, dtype=tf.int32)
        pred_infects = pred_infects.eval(session=tf.Session()) 

        print("Generating figures...")
        img_nums = random.choices(range(num_slices), k=min(4, num_slices))
        for idx in img_nums :
            figname = 'actualvs.pred-patient{:02d}-frame{:03d}.pdf'.format(patient_id, idx)
            outfile = os.path.join(args.outdir, figname)

            compare_actual_and_predicted(ccts, clungs, cinfects, cpred_lungs, pred_infects, 
                                         idx, img_size, outfile)

if __name__ == '__main__':
    test_now()
