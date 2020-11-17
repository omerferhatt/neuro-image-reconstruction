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
from tensorflow.keras.layers import BatchNormalization, Reshape, Add
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import MaxPool1D, MaxPool2D, UpSampling2D
from tensorflow.keras.layers import GlobalAvgPool2D, Flatten
from tensorflow.keras.regularizers import L1L2


class EncoderNet(tf.keras.Model):
	def __init__(self, batch_shape=(1, 345, 5), *args, **kwargs):
		super(EncoderNet, self).__init__(*args, **kwargs)
		# Input layer of model
		self.input_encoder = Input(batch_shape=batch_shape, name='en_input')
		self.input_b_norm = BatchNormalization(name='en_input_b_norm')
		# Output layers
		self.out_b_norm = BatchNormalization(name='en_out_b_norm')
		self.out_flatten = Flatten(name='en_out_flatten')
		# Forward prop.
		self.out = self.call(self.input_encoder)
		super(EncoderNet, self).__init__(inputs=self.input_encoder, outputs=self.out,
										 name='encoder_net', *args, **kwargs)

	@staticmethod
	def conv1d_block(x, block_name: str, block_size: int, filters: list, kernel_sizes: list, strides: list,
					 padding=None, activation=None, pooling=None, residual=False):
		if padding is None:
			padding = 'same'
		if activation is None:
			activation = tf.nn.relu
		# Checking parameters
		assert block_size == len(filters)
		assert block_size == len(kernel_sizes)
		assert block_size == len(strides)
		# Layers
		residual_layers = []
		for i in range(block_size):
			x = Conv1D(filters[i], kernel_sizes[i], strides[i], padding, name=block_name + f'_conv1d_{i + 1}')(x)
			x = activation(x)
			x = BatchNormalization(name=block_name + f'_b_norm_{i + 2}')(x)
			if residual and (i == 0 or i == block_size - 1):
				residual_layers.append(x)
		# Adding residual block to first and last
		if residual:
			x = Add(name=block_name + f'_add')(residual_layers)
		# Adding pooling layer to top
		if pooling is not None:
			x = pooling(pool_size=2, strides=2, name=block_name + f'_pooling')(x)
		return x

	def build(self, *args, **kwargs):
		self._is_graph_network = True
		self._init_graph_network(
			inputs=self.input_encoder,
			outputs=self.out
		)

	def call(self, inputs, *args, **kwargs):
		# Input
		x = self.input_b_norm(inputs)
		# Conv1DBlock 1
		x = self.conv1d_block(x, block_name='en_block1', block_size=2,
							  filters=[64, 64], kernel_sizes=[7, 7], strides=[1, 1],
							  padding='same', activation=tf.nn.relu, residual=True, pooling=MaxPool1D)

		# Conv1DBlock 2
		x = self.conv1d_block(x, block_name='en_block2', block_size=2,
							  filters=[128, 128], kernel_sizes=[5, 5], strides=[1, 1],
							  padding='same', activation=tf.nn.relu, residual=True, pooling=MaxPool1D)

		# Conv1DBlock 3
		x = self.conv1d_block(x, block_name='en_block3', block_size=2,
							  filters=[256, 256], kernel_sizes=[5, 5], strides=[1, 1],
							  padding='same', activation=tf.nn.relu, residual=True, pooling=MaxPool1D)

		# Conv1DBlock 4
		x = self.conv1d_block(x, block_name='en_block4', block_size=2,
							  filters=[512, 512], kernel_sizes=[5, 5], strides=[1, 2],
							  padding='same', activation=tf.nn.relu, residual=False, pooling=None)
		# Output layers
		x = self.out_flatten(x)
		x = self.out_b_norm(x)
		return x

	def get_config(self):
		pass


class DecoderNet(tf.keras.Model):
	def __init__(self, batch_shape, *args, **kwargs):
		super(DecoderNet, self).__init__(*args, **kwargs)
		# Input layer of model
		self.input_decoder = Input(batch_shape=batch_shape, name='de_input')
		self.input_b_norm = BatchNormalization(name='de_input_b_norm')
		self.input_reshape = Reshape(target_shape=(4, 4, 640), name='de_input_reshape')
		# Output layers
		self.out_conv = Conv2D(3, kernel_size=3, strides=1, padding='same', name='de_out_conv')
		self.out = self.call(self.input_decoder)
		super(DecoderNet, self).__init__(inputs=self.input_decoder, outputs=self.out,
										 name='decoder_net', *args, **kwargs)

	@staticmethod
	def conv2d_upsample_block(x, block_name: str, block_size: int, filters: list, kernel_sizes: list, strides: list,
							  padding=None, activation=None, upsample=None, residual=False):
		if padding is None:
			padding = 'same'
		if activation is None:
			activation = tf.nn.relu
		# Checking parameters
		assert block_size == len(filters)
		assert block_size == len(kernel_sizes)
		assert block_size == len(strides)
		# Layers
		residual_layers = []
		for i in range(block_size):
			x = Conv2D(filters[i], kernel_sizes[i], strides[i], padding, name=block_name + f'_conv2d_{i + 1}')(x)
			x = activation(x)
			x = BatchNormalization(name=block_name + f'_b_norm_{i + 2}')(x)
			if residual and (i == 0 or i == block_size - 1):
				residual_layers.append(x)
		# Adding residual block to first and last
		if residual:
			x = Add(name=block_name + f'_add')(residual_layers)
		# Adding pooling layer to top
		if upsample is not None:
			x = upsample(name=block_name + f'_upsample')(x)
		return x

	def build(self, *args, **kwargs):
		self._is_graph_network = True
		self._init_graph_network(
			inputs=self.input_decoder,
			outputs=self.out
		)

	def call(self, inputs, *args, **kwargs):
		x = self.input_b_norm(inputs)
		x = self.input_reshape(x)

		# Conv2DBlock 1
		x = self.conv2d_upsample_block(x, block_name='de_block1', block_size=1,
									   filters=[256], kernel_sizes=[3], strides=[1],
									   padding='same', activation=tf.nn.relu, residual=False, upsample=UpSampling2D)

		# Conv2DBlock 2
		x = self.conv2d_upsample_block(x, block_name='de_block2', block_size=1,
									   filters=[128], kernel_sizes=[3], strides=[1],
									   padding='same', activation=tf.nn.relu, residual=False, upsample=UpSampling2D)

		# Conv2DBlock 3
		x = self.conv2d_upsample_block(x, block_name='de_block3', block_size=1,
									   filters=[64], kernel_sizes=[3], strides=[1],
									   padding='same', activation=tf.nn.relu, residual=False, upsample=UpSampling2D)

		# Conv2DBlock 4
		x = self.conv2d_upsample_block(x, block_name='de_block4', block_size=2,
									   filters=[32, 32], kernel_sizes=[3, 3], strides=[1, 1],
									   padding='same', activation=tf.nn.relu, residual=True, upsample=UpSampling2D)

		# Conv2DBlock 5
		x = self.conv2d_upsample_block(x, block_name='de_block5', block_size=2,
									   filters=[16, 16], kernel_sizes=[3, 3], strides=[1, 1],
									   padding='same', activation=tf.nn.relu, residual=True, upsample=UpSampling2D)

		# Conv2DBlock 6
		x = self.conv2d_upsample_block(x, block_name='de_block6', block_size=2,
									   filters=[8, 8], kernel_sizes=[3, 3], strides=[1, 1],
									   padding='same', activation=tf.nn.relu, residual=True, upsample=UpSampling2D)

		x = self.out_conv(x)
		x = tf.nn.tanh(x)
		return x

	def get_config(self):
		pass


class GeneratorNet(tf.keras.Model):
	def __init__(self, batch_shape, *args, **kwargs):
		super(GeneratorNet, self).__init__(*args, **kwargs)
		self.input_generator = Input(batch_shape=batch_shape, name='input_generator')
		self.encoder_net = EncoderNet(batch_shape=batch_shape)
		self.decoder_net = DecoderNet(batch_shape=self.encoder_net.output_shape)

		self.out = self.call(self.input_generator)
		super(GeneratorNet, self).__init__(inputs=self.input_generator, outputs=self.out,
										   name='generator_net', *args, **kwargs)

	def build(self, *args, **kwargs):
		self._is_graph_network = True
		self._init_graph_network(
			inputs=self.input_generator,
			outputs=self.out
		)

	def call(self, inputs, *args, **kwargs):
		x = self.encoder_net(inputs)
		x = self.decoder_net(x)
		return x

	def get_config(self):
		pass


class DiscriminatorNet(tf.keras.Model):
	def __init__(self, input_shape, *args, **kwargs):
		super(DiscriminatorNet, self).__init__(*args, **kwargs)
		self.input_discriminator = Input(shape=input_shape, name='input_decoder')
		# Layer Properties
		self.block_name = 'disc'
		self.block_padding = 'same'
		self.block_activation = tf.nn.leaky_relu
		# Layers
		self.block_conv_1 = Conv2D(32, kernel_size=4, padding=self.block_padding, strides=2,
									name=self.block_name + '_conv_1')
		self.block_b_norm_1 = BatchNormalization(name=self.block_name + '_b_norm_1')

		self.block_conv_2 = Conv2D(64, kernel_size=4, padding=self.block_padding, strides=2,
									name=self.block_name + '_conv_2')
		self.block_b_norm_2 = BatchNormalization(name=self.block_name + '_b_norm_2')

		self.block_conv_3 = Conv2D(128, kernel_size=4, padding=self.block_padding, strides=2,
									name=self.block_name + '_conv_3')
		self.block_b_norm_3 = BatchNormalization(name=self.block_name + '_b_norm_3')

		self.block_conv_4 = Conv2D(256, kernel_size=4, padding=self.block_padding, strides=2,
									name=self.block_name + '_conv_4')
		self.block_b_norm_4 = BatchNormalization(name=self.block_name + '_b_norm_4')

		self.block_conv_5 = Conv2D(512, kernel_size=4, padding=self.block_padding, strides=2,
									name=self.block_name + '_conv_5')
		self.block_b_norm_5 = BatchNormalization(name=self.block_name + '_b_norm_5')

		self.out_flatten = Flatten(name=self.block_name + '_out_flatten')
		self.out_dense = Dense(1, name=self.block_name + '_out_dense')
		self.out = self.call(self.input_discriminator)
		super(DiscriminatorNet, self).__init__(inputs=self.input_discriminator, outputs=self.out,
											   name='discriminator_net', *args, **kwargs)

	@staticmethod
	def conv2d_block(x, block_name: str, block_size: int, filters: list, kernel_sizes: list, strides: list,
					 padding=None, activation=None, pooling=None, residual=False):
		if padding is None:
			padding = 'same'
		if activation is None:
			activation = tf.nn.relu
		# Checking parameters
		assert block_size == len(filters)
		assert block_size == len(kernel_sizes)
		assert block_size == len(strides)
		# Layers
		residual_layers = []
		for i in range(block_size):
			x = Conv2D(filters[i], kernel_sizes[i], strides[i], padding, name=block_name + f'_conv1d_{i + 1}')(x)
			x = activation(x)
			x = BatchNormalization(name=block_name + f'_b_norm_{i + 2}')(x)
			if residual and (i == 0 or i == block_size - 1):
				residual_layers.append(x)
		# Adding residual block to first and last
		if residual:
			x = Add(name=block_name + f'_add')(residual_layers)
		# Adding pooling layer to top
		if pooling is not None:
			x = pooling(pool_size=2, strides=2, name=block_name + f'_pooling')(x)
		return x

	def build(self, *args, **kwargs):
		self._is_graph_network = True
		self._init_graph_network(
			inputs=self.input_discriminator,
			outputs=self.out
		)

	def call(self, inputs, *args, **kwargs):
		x = self.conv2d_block(inputs, block_name='de_block4', block_size=5,
							  filters=[32, 64, 128, 256, 512], kernel_sizes=[4, 4, 4, 4, 4],
							  strides=[2, 2, 2, 2, 2], padding='same', activation=tf.nn.leaky_relu)

		x = self.out_flatten(x)
		x = self.out_dense(x)
		return x

	def get_config(self):
		pass


if __name__ == '__main__':
	# enc_net = EncoderNet(batch_shape=(1, 320, 5))
	# enc_net.summary()
	# dec_net = DecoderNet(batch_shape=(1, 10240))
	# dec_net.summary()
	# gen = GeneratorNet(batch_shape=(1, 320, 5))
	# gen.summary()
	disc = DiscriminatorNet(input_shape=(256, 256, 3))
	disc.summary()
