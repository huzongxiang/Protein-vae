# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:55:10 2022

@author: hzx
"""


from math import perm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, temp=1.0):
        z_mean, z_log_var = inputs
        std = tf.exp(0.5 * z_log_var)
        eps = tf.random.normal(tf.shape(std))
        return z_mean + eps*std*temp


class Encoder(layers.Layer):
    """" convolution layers, downsampling, reduce input dimension to latent dimension """

    def __init__(self, dim_mid=4, dim_dense=512, dim_latent=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sequence = keras.Sequential([

            layers.Input(shape=(512, 512, 1)),

            layers.Conv2D(dim_mid * 2, 3, strides=1, padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2D(dim_mid * 2, 3, strides=2, padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2D(dim_mid * 4, 4, strides=2, padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2D(dim_mid * 8, 3, strides=2, padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2D(dim_mid * 8, 3, strides=1, padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2D(dim_mid * 8, 3, strides=2, padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2D(dim_mid * 16, 3, strides=2, padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2D(dim_mid * 16, 3, strides=1, padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2D(dim_mid * 32, 4, strides=2, padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2D(dim_mid * 32, 3, strides=2, padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2D(dim_dense, 4, strides=1, padding="valid", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
        ])

        # Use FC to convert to dim_latent x 1 x 1
        self.meanlayer = layers.Dense(dim_latent)
        self.varlayer = layers.Dense(dim_latent)
        self.sampling = Sampling()


    def call(self, input):
        output = self.sequence(input)
        z_mean = self.meanlayer(output)
        z_log_var = self.varlayer(output)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """" transposedconvolution layers, upsampling, regain input features from latent variable """

    def __init__(self, dim_mid=4, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sequence = keras.Sequential([

            layers.Conv2DTranspose(dim_mid * 32, (4,1), strides=(2,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 32, (4,1), strides=(2,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 16, (4,1), strides=(2,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 16, (3,1), strides=(1,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 16, (3,1), strides=(1,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 16, (4,1), strides=(2,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 16, (3,1), strides=(1,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 16, (3,1), strides=(1,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 16, (4,1), strides=(2,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 8, (4,1), strides=(2,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 8, (4,1), strides=(2,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 8, (3,1), strides=(1,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 8, (3,1), strides=(1,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 8, (4,1), strides=(2,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(dim_mid * 4, (4,1), strides=(2,1), padding="same", use_bias=True),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),

            layers.Conv2DTranspose(3, (3,1), strides=(1,1), padding="same", use_bias=True),
        ])


    def call(self, input):
        coord = tf.squeeze(self.sequence(input))
        return coord


class Struct2Map(layers.Layer):
    """" calculate inputs coords of backbones to get contact maps """

    def call(self, inputs):
        square = tf.expand_dims(tf.reduce_sum(tf.square(inputs), axis=-1), axis=-1)
        cross = tf.einsum('ijk,ikm->ijm', inputs, tf.transpose(inputs, perm=(0, 2, 1)))
        contact_maps = tf.expand_dims(tf.sqrt(tf.math.abs(square + tf.transpose(square, perm=(0, 2, 1)) - cross*2)), axis=-1)
        return contact_maps


class Dihedral_phi(layers.Layer):
    """" calculate ramachandran angles of backbones """

    def call(self, inputs):
        b_vectors = inputs[:, :, :-1] - inputs[:, :, 1:]

        n2 = tf.linalg.cross(b_vectors[:, :, 1], b_vectors[:, :, 0])
        n0  =tf.linalg.cross(b_vectors[:, :, 1], b_vectors[:, :, 2])

        # Normalize vectors
        n2 = n2/tf.reshape(tf.sqrt(tf.einsum("ijk,ijk->ij", n2, n2)), (tf.shape(inputs)[0], -1, 1))
        n0 = n0/tf.reshape(tf.sqrt(tf.einsum("ijk,ijk->ij", n0, n0)), (tf.shape(inputs)[0], -1, 1))
        cos = tf.einsum("ijk,ijk->ij", n0, n2)
        return cos


class Dihedral_psi(layers.Layer):
    """" calculate ramachandran angles of backbones """

    def call(self, inputs):
        b_vectors = inputs[:, :, :-1] - inputs[:, :, 1:]

        n2 = tf.linalg.cross(b_vectors[:, :, 1], b_vectors[:, :, 0])
        n1  =tf.linalg.cross(b_vectors[:, :, 0], b_vectors[:, :, 2])

        # Normalize vectors
        n2 = n2/tf.reshape(tf.sqrt(tf.einsum("ijk,ijk->ij", n2, n2)), (tf.shape(inputs)[0], -1, 1))
        n1 = n1/tf.reshape(tf.sqrt(tf.einsum("ijk,ijk->ij", n1, n1)), (tf.shape(inputs)[0], -1, 1))
        cos = tf.einsum("ijk,ijk->ij", n1, n2)
        return cos