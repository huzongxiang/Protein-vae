# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:40:49 2022

@author: hzx
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence


class ProteinSequence(Sequence):
    def __init__(self, datas, batch_size):
        self.datas = datas
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.datas) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.datas[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.arra(batch_x)