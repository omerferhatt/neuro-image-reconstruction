#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from data.data_pipeline import Pipeline
from model.net import GeneratorNet, DiscriminatorNet
from model.custom_losses import discriminator_loss, generator_loss

tf.executing_eagerly()


class Train:
    def __init__(self, disc_model: DiscriminatorNet, gen_model: GeneratorNet, data: Pipeline):
        self.pipeline = data
        # Main models
        self.discriminator = disc_model
        self.generator = gen_model
        # Optimizers
        self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.9)
        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9)
        # Loss function
        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss
        # History logs
        self.history = {
            'disc_loss': [],
            'gen_loss': []
        }

    def train_step(self, raw_signal, real_images):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # run the generator with the random noise batch
            eps = tf.random.normal(shape=(1, 256))
            gen_out = self.generator([raw_signal, eps], training=True)

            # run the discriminator with real input images
            d_logits_real = self.discriminator(real_images, training=True)
            # run the discriminator with fake input images (images from the generator)
            d_logits_fake = self.discriminator(gen_out, training=True)

            # compute the generator loss
            gen_loss = self.generator_loss(d_logits_fake)
            # compute the discriminator loss
            disc_loss = self.discriminator_loss(d_logits_real, d_logits_fake)
        g_grads = g_tape.gradient(gen_loss, self.generator.trainable_weights)
        d_grads = d_tape.gradient(disc_loss, self.discriminator.trainable_weights)

        self.discriminator_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        self.generator_opt.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        self.history['disc_loss'].append(disc_loss), self.history['gen_loss'].append(gen_loss)
        return gen_out, disc_loss, gen_loss

    def train(self, epoch, batch_size):
        step_size = self.pipeline.total_record // batch_size
        for e in range(epoch):
            print(f"Epoch: {e+1}")
            for s in range(step_size):
                eeg_signal, image = next(self.pipeline.generator)
                generated_image, d_loss, g_loss = self.train_step(eeg_signal, image)
                if s % 200 == 0 and s != 0:
                    print(f"Step: {s}\t"
                          f"discriminator_loss:{d_loss:.4f}\tgenerator_loss:{g_loss:.4f}\n")
                    self.visualize_result(image, generated_image)

    @staticmethod
    def visualize_result(real, fake):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        temp_image = np.array((real * 127.5) + 127.5, dtype=np.uint8).squeeze()
        temp_generated = np.array((fake * 127.5) + 127.5, dtype=np.uint8).squeeze()
        ax[0].imshow(temp_image)
        ax[1].imshow(temp_generated)
        fig.savefig('train_step.png', dpi=200)
        plt.close('all')
