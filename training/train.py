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
import numpy as np
import tensorflow as tf

from data.data_pipeline import Pipeline
from model.net import AdverserialNet


class Train:
	def __init__(self, model: AdverserialNet, data: Pipeline, epoch=50, batch_size=1):
		self.pipeline = data
		# Training parameters
		self.epoch = epoch
		self.batch_size = batch_size
		self.step_size = self.pipeline.total_record // self.batch_size
		# Main adverserial and sub-models
		self.adverserial = model
		self.discriminator = self.adverserial.discriminator_net
		self.generator = self.adverserial.generator_net
		self.encoder = self.generator.encoder_net
		self.decoder = self.generator.decoder_net
		# Optimizers
		self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
		self.adverserial_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
		# Loss function
		self.loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		# Metrics
		self.discriminator_acc = tf.keras.metrics.BinaryAccuracy()
		self.adversarial_acc = tf.keras.metrics.BinaryAccuracy()
		# History logs
		self.discriminator_hist = []
		self.adversarial_hist = []

	@tf.function
	def discriminator_train_step(self, x, y):
		with tf.GradientTape() as tape:
			disc_logits = self.discriminator(x, training=True)
			loss_value = self.loss_func(y, disc_logits)
		grads = tape.gradient(loss_value, self.discriminator.trainable_weights)
		self.discriminator_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))
		self.discriminator_acc.update_state(y, disc_logits)
		self.discriminator_hist.append([loss_value])
		return loss_value

	@tf.function
	def adversarial_train_step(self, x, y):
		with tf.GradientTape() as tape:
			gen_out = self.generator(x, training=True)
			disc_logits = self.discriminator(gen_out, training=False)
			loss_value = self.loss_func(y, disc_logits)
		grads = tape.gradient(loss_value, self.generator.trainable_weights)
		self.adverserial_opt.apply_gradients(zip(grads, self.generator.trainable_weights))
		self.adversarial_acc.update_state(y, disc_logits)
		self.adversarial_hist.append([loss_value])
		return loss_value

	@tf.function
	def train_on_batch(self, real_image, raw_signal):
		# Creating labels
		labels = tf.concat([tf.ones((self.batch_size, 1)), tf.zeros((real_image.shape[0], 1))], axis=0)
		# Add random noise to the labels for regularization
		labels += 0.05 * tf.random.uniform(labels.shape)
		generated_im = self.generator(raw_signal)
		combined_images = tf.concat([generated_im, real_image], axis=0)
		loss_disc = self.discriminator_train_step(combined_images, labels)
		fake_labels = tf.ones(shape=(self.batch_size, 1))
		loss_gen = self.adversarial_train_step(raw_signal, fake_labels)
		return loss_disc, loss_gen, generated_im

	def train(self):
		for epoch in range(self.epoch):
			print(f"Epoch: {epoch}")
			for step in range(self.step_size):
				eeg_signal, image = next(self.pipeline.generator)
				disc_loss, gen_loss, generated_im = self.train_on_batch(image, eeg_signal)
				if step % 100 == 0 and step != 0:
					print(f"Step: {step}")
					print(f"discriminator_loss:{np.array(disc_loss):.4f}\t generator_loss:{np.array(gen_loss):.4f}\n")
					plt.imshow(generated_im[0, :, :, :])
					plt.show()


if __name__ == '__main__':
	pipeline = Pipeline('data/dataset.csv', shuffle=10)
	adv_net = AdverserialNet(batch_shape=(1, 345, 5))
	trainer = Train(model=adv_net, data=pipeline)
	trainer.train()
