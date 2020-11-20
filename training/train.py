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

tf.executing_eagerly()

from data.data_pipeline import Pipeline
from model.net import GeneratorNet, DiscriminatorNet
from model.custom_losses import discriminator_loss, generator_loss


class Train:
    def __init__(self, disc_model: DiscriminatorNet, gen_model: GeneratorNet,
                 disc_loss, gen_loss,
                 data: Pipeline):
        self.pipeline = data
        # Training parameters
        self.step_size = None
        # Main models
        self.discriminator = disc_model
        self.generator = gen_model
        # Optimizers
        self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.5, beta_2=0.9)
        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        # Loss function
        self.discriminator_loss = disc_loss
        self.generator_loss = gen_loss
        # Metrics
        self.discriminator_acc = tf.keras.metrics.BinaryAccuracy()
        self.adversarial_acc = tf.keras.metrics.BinaryAccuracy()
        # History logs
        self.history = {
            'disc_loss': [],
            'gen_loss': []
        }

    def train_step(self, raw_signal, real_images):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # run the generator with the random noise batch
            gen_out = self.generator(raw_signal, is_training=True)

            # run the discriminator with real input images
            d_logits_real = self.discriminator(real_images, is_training=True)
            # run the discriminator with fake input images (images from the generator)
            d_logits_fake = self.discriminator(gen_out, is_training=True)

            # compute the generator loss
            gen_loss = self.generator_loss(d_logits_fake)
            # compute the discriminator loss
            disc_loss = self.discriminator_loss(d_logits_real, d_logits_fake)
        g_grads = g_tape.gradient(gen_loss, self.generator.trainable_weights)
        d_grads = d_tape.gradient(disc_loss, self.discriminator.trainable_weights)

        self.discriminator_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        self.generator_opt.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        self.history['disc_loss'].append(disc_loss), self.history['gen_loss'].append(gen_loss)
        return gen_out

    def train(self, epoch, batch_size):
        self.step_size = self.pipeline.total_record // batch_size
        for e in range(epoch):
            print(f"Epoch: {e+1}")
            for s in range(self.step_size):
                eeg_signal, image = next(self.pipeline.generator)
                generated_im = self.train_step(eeg_signal, image)
                if s % 200 == 0 and s != 0:
                    print(f"Step: {s}")
                    print(f"discriminator_loss:{self.history['disc_loss'][-1]:.4f}"
                          f"\tgenerator_loss:{self.history['gen_loss'][-1]:.4f}\n")
                    fig, ax = plt.subplots(nrows=1, ncols=2)
                    ax[0].imshow(generated_im[0, :, :, :])
                    ax[1].imshow(image[0, :, :, :])
                    fig.savefig('train_step.png', dpi=600)


if __name__ == '__main__':
    pipeline = Pipeline('data/dataset.csv', shuffle=10)
    gen = GeneratorNet(shape=(320, 5))
    disc = DiscriminatorNet(shape=(256, 256, 3))

    trainer = Train(disc_model=disc, gen_model=gen,
                    disc_loss=discriminator_loss, gen_loss=generator_loss,
                    data=pipeline)
    trainer.train(epoch=25, batch_size=1)
