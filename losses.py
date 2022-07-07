# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:34:58 2022

@author: hzx
"""


from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 