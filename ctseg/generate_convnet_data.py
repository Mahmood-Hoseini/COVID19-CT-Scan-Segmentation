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
    pid = 0
    for patient_dir in patient_dirs:
        ct_files = sorted(glob.glob(os.path.join(patient_dir, '*ct*')))
        cts_all = []
        for ct_file in ct_files :
            cts_all.append(cv.imread(ct_file))

        lung_files = sorted(glob.glob(os.path.join(patient_dir, '*lung*')))
        lungs_all = []
        for lung_file in lung_files :
            lungs_all.append(cv.imread(lung_file))

        infect_files = sorted(glob.glob(os.path.join(patient_dir, '*infect*')))
        infects_all = []
        for infect_file in infect_files :
            infects_all.append(cv.imread(infect_file))

        pid += 1
        fname = 'patient' + id_format.format(pid)
        os.makedirs(os.path.join(output_fodler, fname))
        outfile = os.path.join(output_fodler, fname, 'patient' + id_format.format(pid))
        nib.save(nib.Nifti1Image(np.asarray(cts_all), np.eye(4)), outfile + "-ctscan.nii")
        nib.save(nib.Nifti1Image(np.asarray(plungs_all), np.eye(4)), outfile + "-lung_mask.nii")
        nib.save(nib.Nifti1Image(np.asarray(pinfects_all), np.eye(4)), outfile + "-infection_mask.nii")


def generate_now(args):
    " main function. Reads and converts images (.png, .jpeg, ...) to .nii medical images. "
    glob_search = os.path.join(args.indir, "patient*")
    patient_dirs = glob.glob(glob_search)

    print("Found patient directories:")
    for patient_dir in patient_dirs:
        print(patient_dir)

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
