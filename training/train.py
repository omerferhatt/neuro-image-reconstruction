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

import tensorflow as tf
from model.net import AdverserialNet


class Train:
	def __init__(self, model=AdverserialNet, epoch=50, batch_size=1):
		# Training parameters
		self.epoch = epoch
		self.batch_size = batch_size
		# Main adverserial and sub-models
		self.adverserial = model
		self.discriminator = self.adverserial.discriminator
		self.generator = self.adverserial.generator
		self.encoder = self.generator.encoder
		self.decoder = self.generator.decoder
		# Optimizers
		self.discriminator_opt = None
		self.adverserial_opt = None
		# Loss function
		self.loss_func = None
		# Metrics
		self.discriminator_acc = None
		self.adversarial_acc = None
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
		self.adverserial.opt.apply_gradients(zip(grads, self.generator.trainable_weights))
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
			pass
		# TODO: Add training loop to here
