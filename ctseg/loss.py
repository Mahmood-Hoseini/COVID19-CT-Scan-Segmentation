#!/usr/bin/env python

from __future__ import division, print_function

from keras import backend as K
from keras import losses


def dice(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
        return dice

def soft_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    return (2 * intersection + smooth) / (area_true + area_pred + smooth)
    
def hard_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_sorensen_dice(y_true_int, y_pred_int, axis, smooth)

sorensen_dice = hard_sorensen_dice

def sorensen_dice_loss(y_true, y_pred, weights):
    batch_dice_coefs = soft_sorensen_dice(y_true, y_pred, axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * dice_coefs)

def jaccard(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def jaccard_loss(y_true, y_pred, weights):
    batch_jaccard_coefs = soft_jaccard(y_true, y_pred, axis=[1, 2])
    jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * jaccard_coefs)

def weighted_categorical_crossentropy(y_true, y_pred, weights, epsilon=1e-8):
    ndim = K.ndim(y_pred)
    ncategory = K.int_shape(y_pred)[-1]
    # scale predictions so class probabilities of each pixel sum to 1
    y_pred /= K.sum(y_pred, axis=(ndim-1), keepdims=True)
    y_pred = K.clip(y_pred, epsilon, 1-epsilon)
    w = K.constant(weights) * (ncategory / sum(weights))
    # first, average over all axis except classes
    cross_entropies = -K.mean(y_true * K.log(y_pred), axis=tuple(range(ndim-1)))
    return K.sum(w * cross_entropies)

## see https://github.com/keras-team/keras/issues/9395
def dice_loss(y_true, y_pred):
    loss = 1 - dice(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred, weights): #Binary Cross-Entropy
    loss = (weights[0]*losses.binary_crossentropy(y_true, y_pred) + 
            weights[1]*dice_loss(y_true, y_pred))
    return loss

def tversky_loss(y_true, y_pred):
    alpha, beta  = 0.5, 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred
    p1 = ones-y_pred
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    
    T = K.sum(num/den) 
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) + 
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + 
                     K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1*m2)
    score = (2.*K.sum(w*intersection) + smooth) / (K.sum(w*m1) + K.sum(w*m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    averaged_mask = K.pool2d(y_true, pool_size=(50, 50), strides=(1, 1), 
                             padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5.*K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 0.5*weighted_bce_loss(y_true, y_pred, weight) + 0.5*dice_loss(y_true, y_pred)
    return loss