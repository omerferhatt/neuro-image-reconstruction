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
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, BatchNormalization, AvgPool1D


class DecoderNet(tf.keras.Model):
	def __init__(self, input_shape=(1, 5, 345), **kwargs):
		super(DecoderNet, self).__init__(**kwargs)
		# Input
		self.input_decoder = Input(batch_shape=input_shape, name='input_decoder')
		# ConvBlock 1
		self.block1_name = 'block1'
		self.block1_padding = 'valid'
		self.block1_activation = tf.nn.relu
		self.block1_b_norm_1 = BatchNormalization(name=self.block1_name + '_b_norm_1')

		self.block1_conv_1 = Conv1D(32, kernel_size=7, padding=self.block1_padding, activation=self.block1_activation,
									name=self.block1_name + '_conv1')
		self.block1_b_norm_2 = BatchNormalization(name=self.block1_name + '_b_norm_2')

		self.block1_conv_2 = Conv1D(64, kernel_size=7, padding=self.block1_padding, activation=self.block1_activation,
									name=self.block1_name + '_conv2')
		self.block1_b_norm_3 = BatchNormalization(name=self.block1_name + '_b_norm_3')

		self.block1_conv_3 = Conv1D(64, kernel_size=5, padding=self.block1_padding, activation=self.block1_activation,
									name=self.block1_name + '_conv3')
		self.block1_b_norm_4 = BatchNormalization(name=self.block1_name + '_b_norm_4')

		self.block1_avg_pool = AvgPool1D(pool_size=2, padding=self.block1_padding, name=self.block1_name + '_avg_pool')

		self.block2_name = 'block2'
		# TODO: Add activation and padding type as attr.
		self.block2_b_norm_1 = BatchNormalization(name=self.block2_name + '_b_norm_1')

		self.block2_conv_1 = Conv1D(64, kernel_size=5, padding='same', activation=tf.nn.relu,
									name=self.block2_name + '_conv1')
		self.block2_b_norm_2 = BatchNormalization(name=self.block2_name + '_b_norm_2')

		self.block2_conv_2 = Conv1D(64, kernel_size=5, padding='same', activation=tf.nn.relu,
									name=self.block2_name + '_conv2')
		self.block2_b_norm_3 = BatchNormalization(name=self.block2_name + '_b_norm_3')

		self.block2_conv_3 = Conv1D(128, kernel_size=5, padding='same', activation=tf.nn.relu,
									name=self.block2_name + '_conv3')
		self.block2_b_norm_4 = BatchNormalization(name=self.block2_name + '_b_norm_4')

		self.block2_conv_4 = Conv1D(256, kernel_size=5, padding='same', activation=tf.nn.relu,
									name=self.block2_name + '_conv4')
		self.block2_b_norm_5 = BatchNormalization(name=self.block2_name + '_b_norm_5')

		self.block2_avg_pool = AvgPool1D(pool_size=2, padding='same', name=self.block2_name + '_avg_pool')

		self.out = self.call(self.input_decoder)
		super(DecoderNet, self).__init__(inputs=self.input_decoder, outputs=self.out, **kwargs)

	def build(self):
		self._is_graph_network = True
		self._init_graph_network(
			inputs=self.input_layer,
			outputs=self.out
		)

	def call(self, inputs):
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
		return self.block2_avg_pool(x)


if __name__ == '__main__':
	model = DecoderNet(input_shape=(1, 345, 5))
	model.summary()
