from __future__ import division, print_function

import os, glob, re, tqdm, re
import nibabel as nib
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
from PIL import Image, ImageDraw
import cv2 as cv
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure

from ctseg import opts

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

def crop_(img, boundaries):
    minx, miny, maxx, maxy = boundaries
    return img[miny:miny+maxy, minx:minx+maxx]

def make_lungmask_bbox(image, bimage, display_flag=False):
    height, width = bimage.shape
    _, thresh = cv.threshold(bimage.astype('uint8'), 0.5, 1, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img_cnt = cv.drawContours(bimage, contours, -1, (0,255,0), 3)
    
    if len(contours) < 2 :
        raise Exception("Did not detect both lungs")
        
    x0, y0, w0, h0 = cv.boundingRect(contours[0])
    x1, y1, w1, h1 = cv.boundingRect(contours[1])

    B = [min(x0,x1)-round(0.05*width), 
         min(y0,y1)-round(0.05*height), 
         max(x0+w0,x1+w1)-min(x0,x1)+round(0.1*width), 
         max(y0+h0,y1+h1)-min(y0,y1)+round(0.1*height)]
    B = [max(B[0],0), max(B[1],0), min(B[2], width), min(B[3], height)]
    
    if (B[2]<30 or B[3]<30) :
        raise Exception("bbox doesn't seem to contain much data")
        
    if display_flag:
        cct_img = crop_(image, B)
        rect = patches.Rectangle((B[0],B[1]),B[2],B[3],linewidth=1,
                                        edgecolor='r',facecolor='none')
        fig, ax = plt.subplots(1, 4, figsize=[12, 3])
        ax[0].set_title("Original CT")
        ax[0].imshow(image, cmap='bone')
        ax[0].axis('off')
        ax[1].set_title("binary image (from model)")
        ax[1].imshow(bimage, cmap='bone')
        ax[1].set_title('detected contours (green)')
        ax[1].imshow(img_cnt)
        ax[2].set_title('bounding box')
        ax[2].imshow(image, cmap='bone')
        ax[2].add_patch(rect)
        ax[3].set_title('cropped CT')
        ax[3].imshow(cct_img, cmap='bone')
        plt.show()
        
    return B


def load_single_image(fname, img_size=128) :
    img = nib.load(fname)
    height, width, slices = img.shape
    arr_img = img.get_fdata()
    arr_img = np.rot90(np.array(arr_img))

    # truncate frist and last 20% of the slices
    arr_img = np.reshape(np.rollaxis(arr_img, 2), (slices, height, width, 1))
    sel_slices = range(round(slices*0.2), round(slices*0.8))
    arr_img = arr_img[sel_slices, :, :]

    ## Resizing to make images smaller
    img_all = []
    for ii in range(len(arr_img)) :
        img = cv.resize(arr_img[ii], dsize=(img_size, img_size), 
                                                interpolation=cv.INTER_AREA)
        img = np.reshape(img, (img_size, img_size, 1))
        img_all.append(img)

    return img_all


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

        # load all data
        self.load_images()


    @property
    def patient_id(self) :
        return int(re.findall(r"patient\d\d/", self.ctscan_file)[0][-3:-1])


    def load_images(self):
        # img_size is the preferred image size to which the image is to be resized
        img_size = 128

        print('Loading images...')
        cts_all = load_single_image(self.ctscan_file, img_size=img_size)

        if os.path.isfile(self.lung_mask_file) :
            lungs_all = load_single_image(self.lung_mask_file, img_size=img_size)
        else :
            lungs_all = []

        if os.path.isfile(self.infection_mask_file) :
            infects_all = load_single_image(self.infection_mask_file, img_size=img_size)
        else :
            infects_all = []

        self.cts_all = cts_all
        self.lungs_all = lungs_all
        self.infects_all = infects_all
        self.image_size = img_size


    def write_gif(self, choice_str='cts', fps=10) :
        """Creates a GIF file.
        inputs: 
            - choice_str: either 'cts', 'lungs', or 'infections'
        """
        args = opts.parse_arguments()
        outfile = 'gif-pid{:02d}-{}.gif'.format(self.patient_id, choice_str)
        output_file = os.path.join(args.outdir, outfile)

        image_dims = (self.image_size, self.image_size)
        if choice_str == 'cts' :
            images = self.cts_all
        elif choice_str == 'lungs' :
            images = self.lungs_all
        elif choice_str == 'infections' :
            images = self.infects_all
        else :
            raise Exception("Unknown choice. Your options are 'cts', 'lungs', or 'infections'")      

        fig = plt.figure()
        fig.set_size_inches(image_dims[0]/50, image_dims[1]/50)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Create frames
        print('writing images ...')
        images = np.squeeze(images)
        frames = []
        for ii in range(len(images)):
            plt_im = plt.imshow(images[ii], cmap='bone', vmin=0, vmax=1, animated=True)
            frames.append([plt_im])

        # Save into a GIF file that loops forever
        animation = anim.ArtistAnimation(fig, frames)
        animation.save(output_file, writer='imagemagick', fps=fps)       
        print('Image saved.')
    

    def write_video(self, output_file, choice_str='cts', FPS=24) :
        args = opts.parse_arguments()

        outfile = 'vid-pid{:02d}-{}.mp4'.format(self.patient_id, choice_str)
        output_file = os.path.join(args.outdir, outfile)

        image_dims = (self.image_size, self.image_size)
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

