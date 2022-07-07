# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 21:36:48 2022

@author: huzongxiang
"""


from doctest import OutputChecker
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from layer import Encoder, Decoder, Struct2Map, Dihedral_phi, Dihedral_psi


class VAE(keras.Model):
    """"  trainning model of VAE """

    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.struct2map = Struct2Map()
        self.dihedral_phi = Dihedral_phi()
        self.dihedral_psi = Dihedral_psi()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.phi_loss_tracker = keras.metrics.Mean(
            name="dihedral_reconstruction_loss"
        )
        self.psi_loss_tracker = keras.metrics.Mean(
            name="dihedral_reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.phi_loss_tracker,
            self.psi_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):
        with tf.GradientTape() as tape:

            inputs = tf.reshape(data, (tf.shape(data)[0], -1, 3))
            maps = self.struct2map(inputs)
            z_mean, z_log_var, z = self.encoder(maps)

            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.MAE(inputs, reconstruction), axis=-1)
            )

            outputs = tf.reshape(reconstruction, (tf.shape(data)[0], -1, 4, 3))

            dihedral_phi = self.dihedral_phi(data)
            dihedral_psi = self.dihedral_psi(data)

            reconstructed_phi = self.dihedral_phi(outputs)
            reconstructed_psi = self.dihedral_psi(outputs)

            phi_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.MAE(dihedral_phi, reconstructed_phi), axis=-1))

            psi_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.MAE(dihedral_psi, reconstructed_psi), axis=-1))

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss + phi_loss + psi_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.phi_loss_tracker.update_state(phi_loss)
        self.psi_loss_tracker.update_state(psi_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "phi_loss": self.phi_loss_tracker.result(),
            "psi_loss": self.psi_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }