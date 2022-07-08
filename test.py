# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:27:52 2022

@author: hzx
"""

import tensorflow as tf
from tensorflow import keras
from model import VAE

datas = tf.random.normal([1024, 128, 4, 3], mean=35, stddev=2)

vae = VAE()
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(datas, epochs=10, batch_size=64)