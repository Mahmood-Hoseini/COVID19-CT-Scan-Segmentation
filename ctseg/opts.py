from __future__ import division, print_function

import os
import argparse
import configparser
import logging

definitions = [
    # model               type   default help
    ('model',            (str,   'convnet',          "Model: convnet")),
    ('num_filters',      (list,  [32,128,128,256],   "Number of filters in convolutional layers.")),
    ('padding',          (str,   'same',             "Padding in convolutional layers. Either `same' or `valid'.")),
    ('dropout',          (float,  0.25,              "Rate for dropout of activation units.")),

    # loss
    ('loss',             (str,   'bce_dice', "Loss function: `pixel' for pixel-wise cross entropy, `dice' for dice coefficient.")),
    ('loss-weights',     {'type': float, 'nargs': '+', 'default': [0.5, 0.5],
                          'help': "When using binary-cross-entropy loss"}),
    ('output_weights',   {'type': float, 'nargs': '+', 'default': [0.5, 0.5],
                          'help': "Relative contribution of two outputs (lung_output and infect_output)"}),

    # training
    ('epochs',           (int,   100,    "Number of epochs to train.")),
    ('batch-size',       (int,   128,    "Mini-batch size for training.")),
    ('validation-split', (float, 0.1,    "Percentage of training data to hold out for validation.")),
    ('optimizer',        (str,   'adam', "Optimizer: sgd, rmsprop, adagrad, adadelta, adam, adamax, or nadam.")),
    ('metrics',          (str,   'dice', "Training metrics: dice, jaccard, accuracy")),
    ('learning-rate',    (float, 1e-4,   "Optimizer learning rate.")),
    ('momentum',         (float, None,   "Momentum for SGD optimizer.")),
    ('decay',            (float, None,   "Learning rate decay (not applicable for nadam).")),
    ('shuffle_train_val',{'default': False, 'action': 'store_true',
                          'help': "Shuffle images before splitting into train vs. val."}),
    ('shuffle',          {'default': False, 'action': 'store_true',
                          'help': "Shuffle images before each training epoch."}),
    ('seed',             (int,   42,   "Seed for numpy RandomState")),

    # files
    ('datadir',          (str,   'training-set',       "Directory containing patientXX/ directories.")),
    ('testdir',          (str,   'testing-set',        "Directory containing patientXX/ directories for testing.")),
    ('outdir',           (str,   'outputs',            "Directory to write output data.")),
    ('indir',            (str,   '',                   "Directory that contain data.")),
    ('outfile',          (str,   'weights-final.hdf5', "File to write final model weights.")),
    ('load-weights',     (str,   '',                   "Load model weights from specified file to initialize training.")),
    ('checkpoint',       {'default': True,             'action': 'store_true',
                          'help': "Write model weights after each epoch if validation accuracy improves."}),

    # augmentation
    ('augment-training',   {'default': True, 'action': 'store_true',
                            'help': "Whether to apply image augmentation to training set."}),
    ('augment-validation', {'default': True, 'action': 'store_true',
                            'help': "Whether to apply image augmentation to validation set."}),
    ('rotation-range',     (float, 15,        "Rotation range (0-180 degrees)")),
    ('width-shift-range',  (float, 0.2,       "Width shift range, as a float fraction of the width")),
    ('height-shift-range', (float, 0.2,       "Height shift range, as a float fraction of the height")),
    ('shear-range',        (float, 0.1,       "Shear intensity (in radians)")),
    ('zoom-range',         (float, 0.05,      "Amount of zoom. If a scalar z, zoom in [1-z, 1+z]. Can also pass a pair of floats as the zoom range.")),
    ('normalize',          {'default': False, 'action': 'store_true',
                            'help': "Subtract mean and divide by std dev from each image."}),
]

noninitialized = {
    'learning_rate': 'getfloat',
    'momentum': 'getfloat',
    'decay': 'getfloat',
    'seed': 'getint',
}

def update_arguments(args, default, config, section, key):
    # Point of this function is to update the args Namespace.
    value = config.get(section, key)
    if value == '' or value is None:
        return

    # Command-line arguments override config file values
    if getattr(args, key) != default:
        return

    # Config files always store values as strings -- get correct type
    if isinstance(default, bool):
        value = config.getboolean(section, key)
    elif isinstance(default, int):
        value = config.getint(section, key)
    elif isinstance(default, float):
        value = config.getfloat(section, key)
    elif isinstance(default, str):
        value = config.get(section, key)
    elif isinstance(default, list):
        string = config.get(section, key)
        value = [float(x) for x in string.split()]
    elif default is None:
        getter = getattr(config, noninitialized[key])
        value = getter(section, key)
    setattr(args, key, value)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train CN-Net to segment lung and infection from CT scans.")

    for argname, kwargs in definitions:
        d = kwargs
        if isinstance(kwargs, tuple):
            d = dict(zip(['type', 'default', 'help'], kwargs))
        parser.add_argument('--' + argname, **d)

    # allow user to input configuration file
    parser.add_argument('configfile', nargs='?', type=str, 
                        help="Load options from the config file.")

    args = parser.parse_args()

    if args.configfile:
        logging.info("Loading options from config file: {}".format(args.configfile))
        config = configparser.ConfigParser(
            inline_comment_prefixes=['#', ';'], allow_no_value=True)
        config.read(args.configfile)
        for section in config:
            for key in config[section]:
                if key not in args:
                    raise Exception("Unknown option {} in config file.".format(key))
                update_arguments(args, parser.get_default(key), config, section, key)

    for k, v in vars(args).items():
        logging.info("{:20s} = {}".format(k, v))

    return args
