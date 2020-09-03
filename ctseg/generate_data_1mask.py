from __future__ import division, print_function

import os, glob
import argparse
import cv2 as cv
import nibabel as nib
import numpy as np

from ctseg import patient, opts


def generate_train_test_data(patient_dirs, output_fodler, id_format="{:02d}"):
    """
    This function is to convert raw images(.png, .jpg, .jpeg, ...) into .nii images
    readable to the package. 
    Directory should look like this:
        directory/patient*/
                    *ct*.png        -- CT scan image
                    *lung*.png      -- lung mask image (corresponding to CT images)
                    *infect*.png    -- infections mask image (corresponding to CT images)
    inputs :
        - patient directories
        - output directory to save data 
        - patient id format, default=XX
    outputs :
        - None
    """
    pid = 19
    for patient_dir in patient_dirs:
        ## Saving CT images
        print("processing {}".format(patient_dir))
        ct_files = sorted(glob.glob(os.path.join(patient_dir, '*ct*')))
        cts_all = []
        for ct_file in ct_files :
            img = cv.imread(ct_file)
            img = np.rot90(np.moveaxis(img, 0, 2), -1)
            cts_all.append(img[0, :, :])

        pid += 1
        fname = 'patient' + id_format.format(pid)
        if not os.path.isdir(os.path.join(output_fodler, fname)) :
            os.makedirs(os.path.join(output_fodler, fname))
        outfile = os.path.join(output_fodler, fname, 'patient' + id_format.format(pid))
        cts_all = np.moveaxis(np.asarray(cts_all), 0, 2)
        nib_img = nib.Nifti1Image(cts_all, np.eye(4))
        nib.save(nib_img, outfile + "-ctscan.nii")

        ## Saving MASK files
        mask_files = sorted(glob.glob(os.path.join(patient_dir, '*mask*')))
        if len(mask_files) > 0 :
            lungs_all = []
            infects_all = []
            for mask_file in mask_files :
                img = cv.imread(mask_file)
                img = np.rot90(np.moveaxis(img, 0, 2), -1)
                lmask = np.zeros(img.shape)
                lmask[img==1] = 1
                lungs_all.append(lmask[0, :, :])
                imask = np.zeros(img.shape)
                imask[(img==2) | (img==3)] = 1
                infects_all.append(imask[0, :, :])

            lungs_all = np.moveaxis(np.asarray(lungs_all), 0, 2)
            nib_lungs = nib.Nifti1Image(lungs_all, np.eye(4))
            nib.save(nib_lungs, outfile + "-lung_mask.nii")
            infects_all = np.moveaxis(np.asarray(infects_all), 0, 2)
            nib_infects = nib.Nifti1Image(infects_all, np.eye(4))
            nib.save(nib_infects, outfile + "-infection_mask.nii")
        else :
            print("patient doesn't have mask files. Saved CT images anyway.\n")


def generate_now(args):
    """ main function. 
        There is only one mask file with 1's for lung and 2's and 3's for 
        infections. Both CT and mask images have 3 channels.
        Reads and converts images (.png, .jpeg, ...) to .nii medical images.
    """
    glob_search = os.path.join(args.indir, "patient*")
    patient_dirs = glob.glob(glob_search)

    print("Found {:02d} patient directories.".format(len(patient_dirs)))

    split_index = int((1-args.validation_split) * len(patient_dirs))
    train_dirs = patient_dirs[:split_index]
    test_dirs = patient_dirs[split_index:]
    print("First {} patients used as training set, remaining as test.".format(split_index))

    train_folder = os.path.join(args.outdir, "train")
    if not os.path.isdir(train_folder) :
        os.makedirs(train_folder)

    test_folder = os.path.join(args.outdir, "test")
    if not os.path.isdir(test_folder) :
        os.makedirs(test_folder)

    generate_train_test_data(train_dirs, train_folder, id_format="{:02d}")
    generate_train_test_data(test_dirs, test_folder, id_format="{:02d}")


if __name__ == '__main__':
    args = opts.parse_arguments()
    setattr(args, 'indir', 'train-assets')
    generate_now(args)
