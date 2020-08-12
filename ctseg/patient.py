from __future__ import division, print_function

import os, glob, re, tqdm
import nibabel as nib
import numpy as np
import re
from PIL import Image, ImageDraw
import cv2 as cv

clahe = cv.createCLAHE(clipLimit=3.0)

def clahe_enhancer(img, clahe, axes):
    '''Contrast Limited Adaptive Histogram Equalizer'''
    img = np.uint8(img*255)  
    clahe_img = clahe.apply(img)

    if len(axes) > 0 :    
        axes[0].imshow(img, cmap='bone')
        axes[0].set_title("Original CT scan")
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].imshow(clahe_img, cmap='bone')
        axes[1].set_title("CLAHE Enhanced CT scan")
        axes[1].set_xticks([]); axes[1].set_yticks([])

        if len(axes) > 2 :
            axes[2].hist(img.flatten(), alpha=0.4, label='Original CT scan')
            axes[2].hist(clahe_img.flatten(), alpha=0.4, label="CLAHE Enhanced CT scan")
            plt.legend()
        
    return(clahe_img)

def get_contours(img):
    img = np.uint8(img*255)
    
    kernel = np.ones((3,3),np.float32)/9
    img = cv.filter2D(img, -1, kernel)
    
    ret, thresh = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, 2, 1)
    #Areas = [cv.contourArea(cc) for cc in contours]; print(Areas)
    
    # filter contours that are too large or small
    size = get_size(img)
    contours = [cc for cc in contours if contourOK(cc, size)]
    return contours

def get_size(img):
    ih, iw = img.shape
    return iw * ih

def contourOK(cc, size):
    x, y, w, h = cv.boundingRect(cc)
    if ((w < 50 and h > 150) or (w > 150 and h < 50)) : 
        return False # too narrow or wide is bad
    area = cv.contourArea(cc)
    return area < (size * 0.5) and area > 200

def find_boundaries(img, contours):
    # margin is the minimum distance from the edges of the image, as a fraction
    ih, iw = img.shape
    minx = iw
    miny = ih
    maxx = 0
    maxy = 0

    for cc in contours:
        x, y, w, h = cv.boundingRect(cc)
        if x < minx: minx = x
        if y < miny: miny = y
        if x + w > maxx: maxx = x + w
        if y + h > maxy: maxy = y + h

    return (minx, miny, maxx, maxy)

def crop_(img, boundaries):
    minx, miny, maxx, maxy = boundaries
    return img[miny:maxy, minx:maxx]
    
def crop_img(img, axes) :
    contours = get_contours(img)
    #plt.figure() # uncomment to troubleshoot
    #canvas = np.zeros_like(img)
    #cv.drawContours(canvas , contours, -1, (255, 255, 0), 1)
    #plt.imshow(canvas)
    bounds = find_boundaries(img, contours)
    cropped_img = crop_(img, bounds)

    if len(axes) > 0 :
        axes[0].imshow(img, cmap='bone')
        axes[0].set_title("Original CT scan")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        
        axes[1].imshow(cropped_img, cmap='bone')
        axes[1].set_title("Cropped CT scan")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        
    return cropped_img, bounds


class PatientData(object):
    """Data directory structure (for patient 00):
    directory/
      train_assets/patient00/
        patient00-ctscan.nii
        patient00-lung_mask.nii
        patient00-infection_mask.nii
    """
    def __init__(self, directory):
        self.directory = os.path.normpath(directory)

        glob_search = os.path.join(directory, "patient*-ctscan.nii")
        files = glob.glob(glob_search)
        print(files[0])
        if len(files) == 0:
            raise Exception("Couldn't find image file in {}. "
                            "Wrong directory?".format(directory))
        self.ctscan_file = files[0]
        self.lung_mask_file = files[0].replace('-ctscan', '-lung_mask')
        self.infection_mask_file = files[0].replace('-ctscan', '-infection_mask')
        self.patient_id = int(re.findall(r"patient\d\d/", self.ctscan_file)[0][-3:-1])

        # load all data
        self.load_images()


    def load_images(self):
        cts = nib.load(self.ctscan_file)
        lungs = nib.load(self.lung_mask_file)
        infec = nib.load(self.infection_mask_file)
    
        slices = cts.shape[2]

        arr_cts = cts.get_fdata()
        arr_lungs = lungs.get_fdata()
        arr_infec = infec.get_fdata()

        arr_cts = np.rot90(np.array(arr_cts))
        arr_lungs = np.rot90(np.array(arr_lungs))
        arr_infec = np.rot90(np.array(arr_infec))

        # truncate frist and last 20% of the slices
        arr_cts = arr_cts[:, :, round(slices*0.2):round(slices*0.8)]
        arr_lungs = arr_lungs[:, :, round(slices*0.2):round(slices*0.8)]
        arr_infec = arr_infec[:, :, round(slices*0.2):round(slices*0.8)]

        arr_cts = np.reshape(np.rollaxis(arr_cts, 2), 
                        (arr_cts.shape[2],arr_cts.shape[0],arr_cts.shape[1], 1))
        arr_lungs = np.reshape(np.rollaxis(arr_lungs, 2), 
                        (arr_lungs.shape[2],arr_lungs.shape[0],arr_lungs.shape[1], 1))
        arr_infec = np.reshape(np.rollaxis(arr_infec, 2), 
                        (arr_infec.shape[2],arr_infec.shape[0],arr_infec.shape[1], 1))

        
        cts_all = []
        lungs_all = []
        infects_all = []
        max_w, max_h = 0, 0 #max width and height

        # img_size is the preferred image size to which the image is to be resized
        img_size = 512

        for ii in range(arr_cts.shape[0]):
            img_lungs = cv.resize(arr_lungs[ii], dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
            xmax, xmin = img_lungs.max(), img_lungs.min()
            img_lungs = (img_lungs - xmin)/(xmax - xmin)
            cropped_lungs, bounds = crop_img(img_lungs, [])
            lungs_all.append(cropped_lungs)
        
            h, w = cropped_lungs.shape
            max_h, max_w = max(max_h, h), max(max_w, w)

            img_ct = cv.resize(arr_cts[ii], dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
            xmax, xmin = img_ct.max(), img_ct.min()
            img_ct = (img_ct - xmin)/(xmax - xmin)
            clahe_ct = clahe_enhancer(img_ct, clahe, [])
            cropped_ct = crop_(clahe_ct, bounds)
            cts_all.append(cropped_ct)

            img_infec = cv.resize(arr_infec[ii], dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
            cropped_infec = crop_(img_infec, bounds)
            infects_all.append(cropped_infec)

        ## Resizing to make images smaller
        num_pix = 100
        del_lst = []
        print('Loading images...')
        for ii in tqdm.tqdm(range(len(cts_all))) :
            try :
                cts_all[ii] = cv.resize(cts_all[ii], dsize=(num_pix, num_pix), interpolation=cv.INTER_AREA)
                cts_all[ii] = np.reshape(cts_all[ii], (num_pix, num_pix, 1))

                lungs_all[ii] = cv.resize(lungs_all[ii], dsize=(num_pix, num_pix), interpolation=cv.INTER_AREA)
                lungs_all[ii] = np.reshape(lungs_all[ii], (num_pix, num_pix, 1))

                infects_all[ii] = cv.resize(infects_all[ii], dsize=(num_pix, num_pix), interpolation=cv.INTER_AREA)
                infects_all[ii] = np.reshape(infects_all[ii], (num_pix, num_pix, 1))
            except :
                del_lst.append(ii)
        
        for idx in del_lst[::-1] :
            del cts_all[idx]
            del lungs_all[idx]
            del infects_all[idx]

        self.cts_all = cts_all
        self.lungs_all = lungs_all
        self.infects_all = infects_all
        self.image_height, self.image_width = max_h, max_w
        self.image_size = num_pix

            
    def write_video(self, output_file, choice_str='cts', FPS=24):
        image_dims = (self.image_width, self.image_height)
        video = cv.VideoWriter(output_file, -1, FPS, image_dims)
        if choice_str == 'cts' :
            images = self.cts_all
        elif choice_str == 'lungs' :
            images = self.lungs_all
        elif choice_str == 'infections' :
            images = self.infects_all
        else :
            raise Exception("Unknown choice. Your options are 'cts', 'lungs', or 'infections'")

        print('writing images ...')
        for image in self.cts_all:
            grayscale = np.asarray(image * (255/image.max()), dtype='uint8')
            video.write(cv.cvtColor(grayscale, cv.COLOR_GRAY2BGR))
        video.release()
        print('video released.')

