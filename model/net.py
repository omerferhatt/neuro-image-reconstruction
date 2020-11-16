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
from tensorflow.keras.layers import BatchNormalization, Reshape
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, Add
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAvgPool1D, GlobalAvgPool2D, AvgPool1D, MaxPool2D, Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import L1L2


class EncoderNet(tf.keras.Model):
	def __init__(self, batch_shape=(1, 5, 345), *args, **kwargs):
		super(EncoderNet, self).__init__(*args, **kwargs)
		# Input layer of model
		self.input_encoder = Input(batch_shape=batch_shape, name='input_encoder')
		# ConvBlock 1
		self.block1_name = 'en_block1'
		self.block1_padding = 'valid'
		self.block1_activation = tf.nn.relu
		# Layers
		self.block1_b_norm_1 = BatchNormalization(name=self.block1_name + '_b_norm_1')
		self.block1_conv_1 = Conv1D(32, kernel_size=7, strides=2, padding=self.block1_padding,
									activation=self.block1_activation,
									name=self.block1_name + '_conv1')
		self.block1_b_norm_2 = BatchNormalization(name=self.block1_name + '_b_norm_2')
		self.block1_conv_2 = Conv1D(32, kernel_size=7, padding=self.block1_padding, activation=self.block1_activation,
									name=self.block1_name + '_conv2')
		self.block1_b_norm_3 = BatchNormalization(name=self.block1_name + '_b_norm_3')
		self.block1_conv_3 = Conv1D(32, kernel_size=5, padding=self.block1_padding, activation=self.block1_activation,
									name=self.block1_name + '_conv3')
		self.block1_b_norm_4 = BatchNormalization(name=self.block1_name + '_b_norm_4')
		self.block1_avg_pool = AvgPool1D(pool_size=2, strides=2, name=self.block1_name + '_avg_pool')

		# ConvBlock 2
		self.block2_name = 'en_block2'
		self.block2_padding = 'valid'
		self.block2_activation = tf.nn.relu
		# Layers
		self.block2_b_norm_1 = BatchNormalization(name=self.block2_name + '_b_norm_1')
		self.block2_conv_1 = Conv1D(64, kernel_size=5, padding=self.block2_padding, activation=self.block2_activation,
									name=self.block2_name + '_conv1')
		self.block2_b_norm_2 = BatchNormalization(name=self.block2_name + '_b_norm_2')
		self.block2_conv_2 = Conv1D(64, kernel_size=5, padding=self.block2_padding, activation=self.block2_activation,
									name=self.block2_name + '_conv2')
		self.block2_b_norm_3 = BatchNormalization(name=self.block2_name + '_b_norm_3')
		self.block2_conv_3 = Conv1D(128, kernel_size=5, padding=self.block2_padding, activation=self.block2_activation,
									name=self.block2_name + '_conv3')
		self.block2_b_norm_4 = BatchNormalization(name=self.block2_name + '_b_norm_4')
		self.block2_conv_4 = Conv1D(128, kernel_size=5, padding=self.block2_padding, activation=self.block2_activation,
									name=self.block2_name + '_conv4')
		self.block2_b_norm_5 = BatchNormalization(name=self.block2_name + '_b_norm_5')
		self.block2_avg_pool = AvgPool1D(pool_size=2, strides=2, name=self.block2_name + '_avg_pool')

		# ConvBlock 3
		self.block3_name = 'en_block3'
		self.block3_padding = 'valid'
		self.block3_activation = tf.nn.relu
		# Layers
		self.block3_b_norm_1 = BatchNormalization(name=self.block3_name + '_b_norm_1')
		self.block3_conv_1 = Conv1D(256, kernel_size=5, padding=self.block3_padding, activation=self.block3_activation,
									name=self.block3_name + '_conv1')
		self.block3_b_norm_2 = BatchNormalization(name=self.block3_name + '_b_norm_2')
		self.block3_conv_2 = Conv1D(256, kernel_size=5, padding=self.block3_padding, activation=self.block3_activation,
									name=self.block3_name + '_conv2')
		self.block3_b_norm_3 = BatchNormalization(name=self.block3_name + '_b_norm_3')
		self.block3_conv_3 = Conv1D(256, kernel_size=5, padding=self.block3_padding, activation=self.block3_activation,
									name=self.block3_name + '_conv3')
		self.block3_b_norm_4 = BatchNormalization(name=self.block3_name + '_b_norm_4')
		self.block3_avg_pool = AvgPool1D(pool_size=2, strides=2, name=self.block3_name + '_avg_pool')

		# ConvBlock 4
		self.block4_name = 'en_block4'
		self.block4_padding = 'valid'
		self.block4_activation = tf.nn.relu
		# Layers
		self.block4_b_norm_1 = BatchNormalization(name=self.block4_name + '_b_norm_1')
		self.block4_conv_1 = Conv1D(256, kernel_size=3, padding=self.block4_padding, activation=self.block4_activation,
									name=self.block4_name + '_conv1')
		self.block4_b_norm_2 = BatchNormalization(name=self.block4_name + '_b_norm_2')
		self.block4_conv_2 = Conv1D(512, kernel_size=3, padding=self.block4_padding, activation=self.block4_activation,
									name=self.block4_name + '_conv2')
		self.block4_b_norm_3 = BatchNormalization(name=self.block4_name + '_b_norm_3')
		self.block4_conv_3 = Conv1D(512, kernel_size=3, padding=self.block4_padding, activation=self.block4_activation,
									name=self.block4_name + '_conv3')
		self.block4_b_norm_4 = BatchNormalization(name=self.block4_name + '_b_norm_4')

		# Output
		self.block4_flatten = Flatten(name=self.block4_name + '_flatten')
		self.block4_dense_1 = Dense(256, activation=self.block4_activation, name=self.block4_name + '_dense_1',
									kernel_regularizer=L1L2(l1=0.001, l2=0.001))
		self.block4_reshape = Reshape(target_shape=(16, 16, 1), name=self.block4_name + '_reshape')
		# Forward prop.
		self.out = self.call(self.input_encoder)
		super(EncoderNet, self).__init__(inputs=self.input_encoder, outputs=self.out,
										 name='encoder_net', *args, **kwargs)

	def build(self, *args, **kwargs):
		self._is_graph_network = True
		self._init_graph_network(
			inputs=self.input_encoder,
			outputs=self.out
		)

	def call(self, inputs, *args, **kwargs):
		# ConvBlock 1
		x = self.block1_b_norm_1(inputs)
		x = self.block1_conv_1(x)
		x = self.block1_b_norm_2(x)
		x = self.block1_conv_2(x)
		x = self.block1_b_norm_3(x)
		x = self.block1_conv_3(x)
		x = self.block1_b_norm_4(x)
		x = self.block1_avg_pool(x)

		# ConvBlock 2
		x = self.block2_b_norm_1(x)
		x = self.block2_conv_1(x)
		x = self.block2_b_norm_2(x)
		x = self.block2_conv_2(x)
		x = self.block2_b_norm_3(x)
		x = self.block2_conv_3(x)
		x = self.block2_b_norm_4(x)
		x = self.block2_conv_4(x)
		x = self.block2_b_norm_5(x)
		x = self.block2_avg_pool(x)

		# ConvBlock 3
		x = self.block3_b_norm_1(x)
		x = self.block3_conv_1(x)
		x = self.block3_b_norm_2(x)
		x = self.block3_conv_2(x)
		x = self.block3_b_norm_3(x)
		x = self.block3_conv_3(x)
		x = self.block3_b_norm_4(x)
		x = self.block3_avg_pool(x)

		# ConvBlock 4
		x = self.block4_b_norm_1(x)
		x = self.block4_conv_1(x)
		x = self.block4_b_norm_2(x)
		x = self.block4_conv_2(x)
		x = self.block4_b_norm_3(x)
		x = self.block4_conv_3(x)
		x = self.block4_b_norm_4(x)
		x = self.block4_flatten(x)
		x = self.block4_dense_1(x)
		return self.block4_reshape(x)

	def get_config(self):
		pass


class DecoderNet(tf.keras.Model):
	def __init__(self, batch_shape, *args, **kwargs):
		super(DecoderNet, self).__init__(*args, **kwargs)
		# Input layer of model
		self.input_decoder = Input(batch_shape=batch_shape, name='input_decoder')

		# DeConvBlock 1
		self.block1_name = 'de_block1'
		self.block1_padding = 'same'
		self.block1_activation = tf.nn.relu
		# Layers
		self.block1_b_norm_1 = BatchNormalization(name=self.block1_name + '_b_norm_1')
		self.block1_deconv_1 = Conv2D(64, kernel_size=3, padding=self.block1_padding,
									  activation=self.block1_activation,
									  name=self.block1_name + '_deconv1')
		self.block1_b_norm_2 = BatchNormalization(name=self.block1_name + '_b_norm_2')
		self.block1_deconv_2 = Conv2D(64, kernel_size=3, padding=self.block1_padding,
									  activation=self.block1_activation,
									  name=self.block1_name + '_deconv2')
		self.block1_b_norm_3 = BatchNormalization(name=self.block1_name + '_b_norm_3')
		self.block1_add = Add(name=self.block1_name + '_add')
		self.block1_deconv_3 = Conv2DTranspose(64, kernel_size=3, strides=2, padding=self.block1_padding,
											   activation=self.block1_activation,
											   name=self.block1_name + '_deconv3')

		# DeConvBlock 2
		self.block2_name = 'de_block2'
		self.block2_padding = 'same'
		self.block2_activation = tf.nn.relu
		# Layers
		self.block2_b_norm_1 = BatchNormalization(name=self.block2_name + '_b_norm_1')
		self.block2_deconv_1 = Conv2D(128, kernel_size=5, padding=self.block2_padding,
									  activation=self.block2_activation,
									  name=self.block2_name + '_deconv1')
		self.block2_b_norm_2 = BatchNormalization(name=self.block2_name + '_b_norm_2')
		self.block2_deconv_2 = Conv2D(128, kernel_size=3, padding=self.block2_padding,
									  activation=self.block2_activation,
									  name=self.block2_name + '_deconv2')
		self.block2_b_norm_3 = BatchNormalization(name=self.block2_name + '_b_norm_3')
		self.block2_add = Add(name=self.block2_name + '_add')
		self.block2_deconv_3 = Conv2DTranspose(128, kernel_size=3, strides=2, padding=self.block2_padding,
											   activation=self.block2_activation,
											   name=self.block2_name + '_deconv3')

		# DeConvBlock 3
		self.block3_name = 'de_block3'
		self.block3_padding = 'same'
		self.block3_activation = tf.nn.relu
		# Layers
		self.block3_b_norm_1 = BatchNormalization(name=self.block3_name + '_b_norm_1')
		self.block3_deconv_1 = Conv2D(64, kernel_size=5, padding=self.block3_padding,
									  activation=self.block3_activation,
									  name=self.block3_name + '_deconv1')
		self.block3_b_norm_2 = BatchNormalization(name=self.block3_name + '_b_norm_2')
		self.block3_deconv_2 = Conv2D(64, kernel_size=3, padding=self.block3_padding,
									  activation=self.block3_activation,
									  name=self.block3_name + '_deconv2')
		self.block3_b_norm_3 = BatchNormalization(name=self.block3_name + '_b_norm_3')
		self.block3_add = Add(name=self.block3_name + '_add')
		self.block3_deconv_3 = Conv2DTranspose(64, kernel_size=3, strides=2, padding=self.block3_padding,
											   activation=self.block3_activation,
											   name=self.block3_name + '_deconv3')

		# MixBlock
		self.block4_name = 'mix_block'
		self.block4_padding = 'same'
		self.block4_activation = tf.nn.relu
		# Layers
		self.block4_b_norm_1 = BatchNormalization(name=self.block4_name + '_b_norm_1')
		self.block4_deconv_1 = Conv2D(32, kernel_size=5, padding=self.block4_padding,
									  activation=self.block4_activation,
									  name=self.block4_name + '_deconv1')
		self.block4_b_norm_2 = BatchNormalization(name=self.block4_name + '_b_norm_2')
		self.block4_deconv_2 = Conv2D(32, kernel_size=3, padding=self.block4_padding,
									  activation=self.block4_activation,
									  name=self.block4_name + '_deconv2')
		self.block4_b_norm_3 = BatchNormalization(name=self.block4_name + '_b_norm_3')
		self.block4_add = Add(name=self.block4_name + '_add')
		self.block4_deconv_3 = Conv2DTranspose(32, kernel_size=3, strides=2, padding=self.block4_padding,
											   activation=self.block4_activation,
											   name=self.block4_name + '_deconv3')
		self.block4_conv = Conv2D(3, kernel_size=7, padding=self.block4_padding, activation=tf.nn.tanh,
								  name=self.block4_name + '_conv')

		self.out = self.call(self.input_decoder)
		super(DecoderNet, self).__init__(inputs=self.input_decoder, outputs=self.out,
										 name='decoder_net', *args, **kwargs)

	def build(self, *args, **kwargs):
		self._is_graph_network = True
		self._init_graph_network(
			inputs=self.input_decoder,
			outputs=self.out
		)

	def call(self, inputs, *args, **kwargs):
		# DeConvBlock 1
		x = self.block1_b_norm_1(inputs)
		x = self.block1_deconv_1(x)
		x_b1_add = self.block1_b_norm_2(x)
		x = self.block1_deconv_2(x_b1_add)
		x = self.block1_b_norm_3(x)
		x = self.block1_add([x, x_b1_add])
		x = self.block1_deconv_3(x)

		# DeConvBlock 2
		x = self.block2_b_norm_1(x)
		x = self.block2_deconv_1(x)
		x_b2_add = self.block2_b_norm_2(x)
		x = self.block2_deconv_2(x_b2_add)
		x = self.block2_b_norm_3(x)
		x = self.block2_add([x, x_b2_add])
		x = self.block2_deconv_3(x)

		# DeConvBlock 3
		x = self.block3_b_norm_1(x)
		x = self.block3_deconv_1(x)
		x_b3_add = self.block3_b_norm_2(x)
		x = self.block3_deconv_2(x_b3_add)
		x = self.block3_b_norm_3(x)
		x = self.block3_add([x, x_b3_add])
		x = self.block3_deconv_3(x)

		# MixBlock
		x = self.block4_b_norm_1(x)
		x = self.block4_deconv_1(x)
		x_b4_add = self.block4_b_norm_2(x)
		x = self.block4_deconv_2(x_b4_add)
		x = self.block4_b_norm_3(x)
		x = self.block4_add([x, x_b4_add])
		x = self.block4_deconv_3(x)
		x = self.block4_conv(x)
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
	def __init__(self, batch_shape, *args, **kwargs):
		super(DiscriminatorNet, self).__init__(*args, **kwargs)
		self.input_discriminator = Input(batch_shape=batch_shape, name='input_decoder')
		# ConvBlock 1
		self.block1_name = 'disc_block1'
		self.block1_padding = 'same'
		self.block1_activation = tf.nn.relu
		# Layers
		self.block1_b_norm_1 = BatchNormalization(name=self.block1_name + '_b_norm_1')
		self.block1_conv_1 = Conv2D(64, kernel_size=3, padding=self.block1_padding,
									activation=self.block1_activation,
									name=self.block1_name + '_conv1')
		self.block1_b_norm_2 = BatchNormalization(name=self.block1_name + '_b_norm_2')
		self.block1_conv_2 = Conv2D(64, kernel_size=3, padding=self.block1_padding, activation=self.block1_activation,
									name=self.block1_name + '_conv2')
		self.block1_b_norm_3 = BatchNormalization(name=self.block1_name + '_b_norm_3')
		self.block1_conv_3 = Conv2D(64, kernel_size=4, padding=self.block1_padding, activation=self.block1_activation,
									name=self.block1_name + '_conv3')
		self.block1_b_norm_4 = BatchNormalization(name=self.block1_name + '_b_norm_4')
		self.block1_add = Add(name=self.block1_name + '_add')
		self.block1_max_pool = MaxPool2D(pool_size=2, strides=2, name=self.block1_name + '_max_pool')

		# ConvBlock 2
		self.block2_name = 'disc_block2'
		self.block2_padding = 'same'
		self.block2_activation = tf.nn.relu
		# Layers
		self.block2_b_norm_1 = BatchNormalization(name=self.block2_name + '_b_norm_1')
		self.block2_conv_1 = Conv2D(128, kernel_size=4, padding=self.block2_padding,
									activation=self.block2_activation,
									name=self.block2_name + '_conv1')
		self.block2_b_norm_2 = BatchNormalization(name=self.block2_name + '_b_norm_2')
		self.block2_conv_2 = Conv2D(128, kernel_size=3, padding=self.block2_padding, activation=self.block2_activation,
									name=self.block2_name + '_conv2')
		self.block2_b_norm_3 = BatchNormalization(name=self.block2_name + '_b_norm_4')
		self.block2_add = Add(name=self.block2_name + '_add')
		self.block2_max_pool = MaxPool2D(pool_size=2, strides=2, name=self.block2_name + '_max_pool')

		# ConvBlock 3
		self.block3_name = 'disc_block3'
		self.block3_padding = 'same'
		self.block3_activation = tf.nn.relu
		# Layers
		self.block3_b_norm_1 = BatchNormalization(name=self.block3_name + '_b_norm_1')
		self.block3_conv_1 = Conv2D(256, kernel_size=4, padding=self.block3_padding,
									activation=self.block3_activation,
									name=self.block3_name + '_conv1')
		self.block3_b_norm_2 = BatchNormalization(name=self.block3_name + '_b_norm_2')
		self.block3_conv_2 = Conv2D(256, kernel_size=3, padding=self.block3_padding, activation=self.block3_activation,
									name=self.block3_name + '_conv2')
		self.block3_b_norm_3 = BatchNormalization(name=self.block3_name + '_b_norm_4')
		self.block3_add = Add(name=self.block3_name + '_add')
		self.block3_max_pool = MaxPool2D(pool_size=2, strides=2, name=self.block3_name + '_max_pool')

		# ConvBlock 4
		self.block4_name = 'disc_block4'
		self.block4_padding = 'same'
		self.block4_activation = tf.nn.relu
		# Layers
		self.block4_b_norm_1 = BatchNormalization(name=self.block4_name + '_b_norm_1')
		self.block4_conv_1 = Conv2D(512, kernel_size=5, padding=self.block4_padding, activation=self.block4_activation,
									name=self.block4_name + '_conv1')
		self.block4_b_norm_2 = BatchNormalization(name=self.block4_name + '_b_norm_2')
		self.block4_global_avg = GlobalAvgPool2D(name=self.block4_name + '_global_avg')

		# DenseBlock
		self.block5_name = 'disc_block5'
		self.block5_activation = tf.nn.relu
		# Layers
		self.block5_b_norm = BatchNormalization(name=self.block5_name + '_b_norm_1')
		self.block5_dense_1 = Dense(256, activation=self.block5_activation, kernel_regularizer=L1L2(l1=0.002, l2=0.002),
									name=self.block5_name + '_dense_1')
		self.block5_dense_2 = Dense(1, activation=tf.nn.sigmoid, name=self.block5_name + '_dense_2')

		self.out = self.call(self.input_discriminator)
		super(DiscriminatorNet, self).__init__(inputs=self.input_discriminator, outputs=self.out,
											   name='discriminator_net', *args, **kwargs)

	def build(self, *args, **kwargs):
		self._is_graph_network = True
		self._init_graph_network(
			inputs=self.input_discriminator,
			outputs=self.out
		)

	def call(self, inputs, *args, **kwargs):
		# ConvBlock 1
		x = self.block1_b_norm_1(inputs)
		x = self.block1_conv_1(x)
		x_b1_add = self.block1_b_norm_2(x)
		x = self.block1_conv_2(x_b1_add)
		x = self.block1_b_norm_3(x)
		x = self.block1_conv_3(x)
		x = self.block1_b_norm_4(x)
		x = self.block1_add([x, x_b1_add])
		x = self.block1_max_pool(x)

		# ConvBlock 2
		x = self.block2_b_norm_1(x)
		x = self.block2_conv_1(x)
		x_b2_add = self.block2_b_norm_2(x)
		x = self.block2_conv_2(x_b2_add)
		x = self.block2_b_norm_3(x)
		x = self.block2_add([x, x_b2_add])
		x = self.block2_max_pool(x)

		# ConvBlock 3
		x = self.block3_b_norm_1(x)
		x = self.block3_conv_1(x)
		x_b3_add = self.block3_b_norm_2(x)
		x = self.block3_conv_2(x_b3_add)
		x = self.block3_b_norm_3(x)
		x = self.block3_add([x, x_b3_add])
		x = self.block3_max_pool(x)

		# ConvBlock 4
		x = self.block4_b_norm_1(x)
		x = self.block4_conv_1(x)
		x = self.block4_b_norm_2(x)
		x = self.block4_global_avg(x)

		# DenseBlock
		x = self.block5_b_norm(x)
		x = self.block5_dense_1(x)
		x = self.block5_dense_2(x)
		return x

	def get_config(self):
		pass


class AdverserialNet(tf.keras.Model):
	def __init__(self, batch_shape, *args, **kwargs):
		super(AdverserialNet, self).__init__(*args, **kwargs)
		# Input Layer
		self.input_adverserial = Input(batch_shape=batch_shape, name='input_adverserial')
		# Generator and Discriminator net
		self.generator_net = GeneratorNet(batch_shape=self.input_adverserial.shape)
		self.discriminator_net = DiscriminatorNet(batch_shape=self.generator_net.output_shape)

		self.out = self.call(self.input_adverserial)
		super(AdverserialNet, self).__init__(inputs=self.input_adverserial, outputs=self.out,
											 name='adverserial_net', *args, **kwargs)

	def build(self, *args, **kwargs):
		self._is_graph_network = True
		self._init_graph_network(
			inputs=self.input_adverserial,
			outputs=self.out
		)

	def call(self, inputs, *args, **kwargs):
		fake_x = self.generator_net(inputs)
		prediction = self.discriminator_net(fake_x)
		return prediction

	def get_config(self):
		pass


if __name__ == '__main__':
	enc_net = EncoderNet(batch_shape=(1, 345, 5))
	enc_net.summary()
	# adv_net = AdverserialNet(batch_shape=(1, 345, 5))
	# adv_net.summary()
