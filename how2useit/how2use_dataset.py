from __future__ import division, print_function

import os
import numpy as np
from math import ceil

from ctseg import dataset

def img_generator(data_dir, batch_size, val_split):
    (train_gen, train_steps_per_epoch,
    val_gen, val_steps_per_epoch) = dataset.create_generators(data_dir, 
                                                              batch_size,
                                                              img_data=[],
                                                              validation_split=val_split)

    cts, masks = next(train_gen)
    assert cts[0].shape == (128, 128, 1)

    cts, masks = next(val_gen)
    assert cts[0].shape == (128, 128, 1)


## test case
data_dir = "../testing-dir/"
batch_size = 16
val_split = 0.5
img_generator(data_dir, batch_size, val_split)
print('Everything looks good!')
