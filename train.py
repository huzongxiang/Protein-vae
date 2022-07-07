"""
Created on Thu Jul  6 11:27:52 2022

@author: hzx
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import VAE
from process_pbd import load_data


data_path = "./"
datas = np.array(load_data(data_path))

vae = VAE()
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2))
vae.fit(datas, epochs=10, batch_size=32)