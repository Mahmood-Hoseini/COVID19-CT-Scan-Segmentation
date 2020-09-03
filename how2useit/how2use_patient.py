from __future__ import division, print_function

import os, random
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings('ignore')


from ctseg import patient


def test_patient_data(fpath):
    p = patient.PatientData(fpath)
    print('CT scan file loaded: {}'.format(p.ctscan_file))
    print('CT scan tensor shape: {}'.format(p.cts_all[0].shape))
    print('Lung mask tensor shape: {}'.format(p.lungs_all[0].shape))
    print('Infection mask tensor shape: {}'.format(p.infects_all[0].shape))
    assert p.cts_all[0].shape == p.lungs_all[0].shape
    assert p.cts_all[0].shape == p.infects_all[0].shape
    assert p.cts_all[0].shape[0] == p.image_size
    print('Patient ID: {:02d}'.format(p.patient_id))
    print('Successfully loaded and everything checks!')

    return p


def plot_and_save (patient_obj, num_plots=6) :
    slices = len(patient_obj.cts_all)
    r = random.choices(range(slices), k=num_plots)

    fig, axes = plt.subplots(3, num_plots, figsize=(3*num_plots,6))    
    for ii in range(num_plots):
        axes[0,ii].imshow(patient_obj.cts_all[r[ii]][:, :, 0], cmap='bone')
        axes[0,ii].set_title("CT scan")
        axes[0,ii].set_xticks([]); 
        axes[0,ii].set_yticks([])
        
        axes[1,ii].imshow(patient_obj.lungs_all[r[ii]][:, :, 0], cmap='Greens')
        axes[1,ii].set_title("Lungs mask")
        axes[1,ii].set_xticks([]); 
        axes[1,ii].set_yticks([])

        axes[2,ii].imshow(patient_obj.infects_all[r[ii]][:, :, 0], cmap='Reds')
        axes[2,ii].set_title("Infection mask")
        axes[2,ii].set_xticks([]); 
        axes[2,ii].set_yticks([])
    plt.savefig(os.path.join(os.getcwd(), '../outputs/test_patient.pdf'), bbox_inches='tight')


def test_write_video(patient_obj):
    outfile = "../outputs/test_video.mp4"
    path = os.path.join(os.getcwd(), outfile)
    patient_obj.write_video(path)


fpath = "../testing-dir/patient23/"
patient_obj = test_patient_data(fpath)
plot_and_save(patient_obj)
test_write_video(patient_obj)

