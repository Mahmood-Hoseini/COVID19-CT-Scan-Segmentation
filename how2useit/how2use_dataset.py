from __future__ import division, print_function

import os
import numpy as np
from math import ceil

from ctseg import dataset

def img_generator(data_dir, batch_size, val_split):
    (train_gen, train_steps_per_epoch,
    val_gen, val_steps_per_epoch) = dataset.create_generators(data_dir, 
                                                              batch_size,
                                                              val_split)

    cts, lungs_infects_mask_dict = next(train_gen)
    assert cts[0].shape == (100, 100, 1)
    assert 'lung_output' in lungs_infects_mask_dict.keys()
    assert 'infect_output' in lungs_infects_mask_dict.keys()
    assert train_steps_per_epoch == ceil((1-val_split)*55/batch_size)

    cts, lungs_infects_mask_dict = next(val_gen)
    assert cts[0].shape == (100, 100, 1)
    assert val_steps_per_epoch == ceil(val_split*55/batch_size)


## test case
data_dir = "../test-assets/"
batch_size = 16
val_split = 0.5
img_generator(data_dir, batch_size, val_split)
print('Everything looks good!')
