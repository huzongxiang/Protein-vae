# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:54:19 2022

@author: hzx
"""


import tensorflow as tf


# difine a decorator for check Nan value of a tensor
def check_err_value(func):
    def wrapper(*args, **kwargs):
        outputs = func(*args, **kwargs)
        tf.compat.v1.check_numerics(outputs, "non number")
    return wrapper

