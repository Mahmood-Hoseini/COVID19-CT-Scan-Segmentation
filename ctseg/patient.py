from __future__ import division, print_function

import os, glob, re, tqdm, re
import nibabel as nib
import numpy as np
import matplotlib.pylab as plt
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
    return img[minx:maxx, miny:maxy]
    
def make_lungmask(img, img_size=512, display_flag=False):
    height, width = img.shape
    img = (img-np.mean(img))/np.std(img)

    middle = img[int(width/5):int(width/5*4),int(height/5):int(height/5*4)] 
    mean = np.mean(middle)  
    imgmax = np.max(img)
    imgmin = np.min(img)
    
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==imgmax]=mean
    img[img==imgmin]=mean
    
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.
    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    try :
        Lung1 = regions[1].bbox
        Lung2 = regions[2].bbox
        bounds = [min(Lung1[0], Lung2[0]), min(Lung1[1], Lung2[1]),
                  max(Lung1[2], Lung2[2]), max(Lung1[3], Lung2[3])]
    except :
        bounds = [img_size, img_size, 0, 0]

    if display_flag:
        fig, ax = plt.subplots(1, 5, figsize=[15, 3])
        ax[0].set_title("Original CT")
        ax[0].imshow(img, cmap='bone')
        ax[0].axis('off')
        ax[1].set_title("Threshold")
        ax[1].imshow(thresh_img, cmap='bone')
        ax[1].axis('off')
        ax[2].set_title("After Erosion & Dilation")
        ax[2].imshow(dilation, cmap='bone')
        ax[2].axis('off')
        ax[3].set_title("Colored labels")
        ax[3].imshow(labels)
        ax[3].axis('off')
        ax[4].set_title("Cropped CT")
        ax[4].imshow(crop_(img, bounds), cmap='bone')
        ax[4].axis('off')
        plt.show()

    return bounds


def load_single_image(fname, bounds=[], img_size=512) :
    img = nib.load(fname)
    height, width, slices = img.shape
    arr_img = img.get_fdata()
    arr_img = np.rot90(np.array(arr_img))

    # truncate frist and last 20% of the slices
    arr_img = np.reshape(np.rollaxis(arr_img, 2), (slices, height, width, 1))
    sel_slices = range(round(slices*0.2), round(slices*0.8))
    arr_img = arr_img[sel_slices, :, :]

    ## computing boundaries
    if 'ctscan' in fname :
        print('Determining boundaries...')
        for ii in range(arr_img.shape[0]):
            img = cv.resize(arr_img[ii], dsize=(img_size, img_size), 
                                interpolation=cv.INTER_AREA)
            B = make_lungmask(img, img_size=img_size, display_flag=False)
            bounds[0] = min(bounds[0], B[0])
            bounds[1] = min(bounds[1], B[1])
            bounds[2] = max(bounds[2], B[2])
            bounds[3] = max(bounds[3], B[3])

    img_all = []
    for ii in range(arr_img.shape[0]):
        img = cv.resize(arr_img[ii], dsize=(img_size, img_size), 
                                interpolation=cv.INTER_AREA)
        if 'ctscan' in fname :
            cropped_img = crop_(img, bounds)
            img_all.append(cropped_img/255)
        else :
            cropped_img = crop_(img, bounds)
            img_all.append(cropped_img)

    return img_all, bounds


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
        ## Resizing to make images smaller
        num_pix = 100
            
        # img_size is the preferred image size to which the image is to be resized
        img_size = 512

        cts_all, bounds = load_single_image(self.ctscan_file, bounds=[img_size, img_size, 0, 0])

        if os.path.isfile(self.lung_mask_file) :
            lungs_all, bounds = load_single_image(self.lung_mask_file, bounds=bounds)

        if os.path.isfile(self.infection_mask_file) :
            infects_all, bounds = load_single_image(self.infection_mask_file, bounds=bounds)

        ## Resizing to make images smaller
        num_pix = 100
        del_lst = []
        print('Loading images...')
        for ii in tqdm.tqdm(range(len(cts_all))) :
            try :
                cts_all[ii] = cv.resize(cts_all[ii], dsize=(num_pix, num_pix), 
                                            interpolation=cv.INTER_AREA)
                cts_all[ii] = np.reshape(cts_all[ii], (num_pix, num_pix, 1))

                if os.path.isfile(self.lung_mask_file) :
                    lungs_all[ii] = cv.resize(lungs_all[ii], dsize=(num_pix, num_pix), 
                                                interpolation=cv.INTER_AREA)
                    lungs_all[ii] = np.reshape(lungs_all[ii], (num_pix, num_pix, 1))

                    infects_all[ii] = cv.resize(infects_all[ii], dsize=(num_pix, num_pix), 
                                                interpolation=cv.INTER_AREA)
                    infects_all[ii] = np.reshape(infects_all[ii], (num_pix, num_pix, 1))
            except :
                del_lst.append(ii)
        
        for idx in del_lst[::-1] :
            del cts_all[idx]
            if os.path.isfile(self.lung_mask_file) :
                del lungs_all[idx]
                del infects_all[idx]

        self.cts_all = cts_all
        self.lungs_all = lungs_all
        self.infects_all = infects_all
        self.image_size = num_pix


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

