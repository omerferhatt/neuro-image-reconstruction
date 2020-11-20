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
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import MaxPool1D, UpSampling2D, GlobalAvgPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow_addons.layers import SpectralNormalization

from model.custom_layers import CustomConvBlock, CustomTanH


class EncoderNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(EncoderNet, self).__init__(*args, **kwargs)
        # Input layer of model
        self.block1_conv1d = CustomConvBlock(Conv1D, filters=32, kernel_size=13, stride=1, padding='same',
                                             spectral_norm=False, batch_norm=True, activation=ReLU, name='en_block1')
        self.block1_max_pool = MaxPool1D(name='en_block1_max_pool')

        self.block2_conv1d = CustomConvBlock(Conv1D, filters=64, kernel_size=11, stride=1, padding='same',
                                             spectral_norm=False, batch_norm=True, activation=ReLU, name='en_block2')
        self.block2_max_pool = MaxPool1D(name='en_block2_max_pool')

        self.block3_conv1d = CustomConvBlock(Conv1D, filters=128, kernel_size=9, stride=1, padding='same',
                                             spectral_norm=False, batch_norm=True, activation=ReLU, name='en_block3')
        self.block3_max_pool = MaxPool1D(name='en_block3_max_pool')

        self.block4_conv1d = CustomConvBlock(Conv1D, filters=256, kernel_size=7, stride=1, padding='same',
                                             spectral_norm=False, batch_norm=True, activation=ReLU, name='en_block4')
        self.block4_max_pool = MaxPool1D(name='en_block4_max_pool')

        self.block5_conv1d = CustomConvBlock(Conv1D, filters=512, kernel_size=5, stride=1, padding='same',
                                             spectral_norm=False, batch_norm=True, activation=ReLU, name='en_block5')
        self.block5_flatten = Flatten(name='en_block5_flatten')

    def call(self, inputs, *args, **kwargs):
        # Conv1D Block 1
        x = self.block1_conv1d(inputs)
        x = self.block1_max_pool(x)
        # Conv1D Block 2
        x = self.block2_conv1d(x)
        x = self.block2_max_pool(x)
        # Conv1D Block 3
        x = self.block3_conv1d(x)
        x = self.block3_max_pool(x)
        # Conv1D Block 4
        x = self.block4_conv1d(x)
        x = self.block4_max_pool(x)
        # Conv1D Block 5
        x = self.block5_conv1d(x)
        x = self.block5_flatten(x)
        return x

    def get_config(self):
        pass


class DecoderNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(DecoderNet, self).__init__(*args, **kwargs)
        self.block1_conv2d = CustomConvBlock(Conv2D, filters=256, kernel_size=3, stride=1, padding='same',
                                             spectral_norm=True, batch_norm=True, activation=ReLU, name='de_block1')
        self.block1_up_sample = UpSampling2D(name='de_block1_up_sample')

        self.block2_conv2d = CustomConvBlock(Conv2D, filters=128, kernel_size=3, stride=1, padding='same',
                                             spectral_norm=True, batch_norm=True, activation=ReLU, name='de_block2')
        self.block2_up_sample = UpSampling2D(name='de_block2_up_sample')

        self.block3_conv2d = CustomConvBlock(Conv2D, filters=64, kernel_size=3, stride=1, padding='same',
                                             spectral_norm=True, batch_norm=True, activation=ReLU, name='de_block3')
        self.block3_up_sample = UpSampling2D(name='de_block3_up_sample')

        self.block4_conv2d = CustomConvBlock(Conv2D, filters=32, kernel_size=3, stride=1, padding='same',
                                             spectral_norm=True, batch_norm=True, activation=ReLU, name='de_block4')
        self.block4_up_sample = UpSampling2D(name='de_block4_up_sample')

        self.block5_conv2d = CustomConvBlock(Conv2D, filters=16, kernel_size=3, stride=1, padding='same',
                                             spectral_norm=True, batch_norm=True, activation=ReLU, name='de_block5')
        self.block5_up_sample = UpSampling2D(name='de_block5_up_sample')

        self.block6_conv2d = CustomConvBlock(Conv2D, filters=8, kernel_size=3, stride=1, padding='same',
                                             spectral_norm=True, batch_norm=True, activation=ReLU,
                                             name='de_block6')
        self.block6_up_sample = UpSampling2D(name='de_block6_up_sample')

        self.block7_conv2d = CustomConvBlock(Conv2D, filters=3, kernel_size=4, stride=1, padding='same',
                                             spectral_norm=False, batch_norm=False, activation=CustomTanH,
                                             name='de_block7')

    def call(self, inputs, *args, **kwargs):
        # Conv2D Block 1
        x = self.block1_conv2d(inputs)
        x = self.block1_up_sample(x)
        # Conv2D Block 2
        x = self.block2_conv2d(x)
        x = self.block2_up_sample(x)
        # Conv2D Block 3
        x = self.block3_conv2d(x)
        x = self.block3_up_sample(x)
        # Conv2D Block 4
        x = self.block4_conv2d(x)
        x = self.block4_up_sample(x)
        # Conv2D Block 5
        x = self.block5_conv2d(x)
        x = self.block5_up_sample(x)
        # Conv2D Block 6
        x = self.block6_conv2d(x)
        x = self.block6_up_sample(x)
        # Conv2D Block 7
        x = self.block7_conv2d(x)
        return x

    def get_config(self):
        pass


class GeneratorNet(tf.keras.Model):
    def __init__(self, shape, *args, **kwargs):
        super(GeneratorNet, self).__init__(*args, **kwargs)
        self.input_generator = Input(shape=shape, name='input_generator')
        self.encoder_net = EncoderNet()
        self.reshape = Reshape(target_shape=(4, 4, 640))
        self.decoder_net = DecoderNet()

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
        x = self.reshape(x)
        x = self.decoder_net(x)
        return x

    def get_config(self):
        pass


class DiscriminatorNet(tf.keras.Model):
    def __init__(self, shape, *args, **kwargs):
        super(DiscriminatorNet, self).__init__(*args, **kwargs)
        self.input_discriminator = Input(shape=shape, name='input_discriminator')
        self.block1_conv2d = CustomConvBlock(Conv2D, filters=32, kernel_size=4, stride=2, padding='same',
                                             spectral_norm=True, batch_norm=False, activation=LeakyReLU,
                                             name='disc_block1')
        self.block2_conv2d = CustomConvBlock(Conv2D, filters=64, kernel_size=4, stride=2, padding='same',
                                             spectral_norm=True, batch_norm=False, activation=LeakyReLU,
                                             name='disc_block2')
        self.block3_conv2d = CustomConvBlock(Conv2D, filters=128, kernel_size=4, stride=2, padding='same',
                                             spectral_norm=True, batch_norm=False, activation=LeakyReLU,
                                             name='disc_block3')
        self.block4_conv2d = CustomConvBlock(Conv2D, filters=256, kernel_size=4, stride=2, padding='same',
                                             spectral_norm=True, batch_norm=False, activation=LeakyReLU,
                                             name='disc_block4')
        self.block5_conv2d = CustomConvBlock(Conv2D, filters=512, kernel_size=4, stride=2, padding='same',
                                             spectral_norm=True, batch_norm=False, activation=LeakyReLU,
                                             name='disc_block5')
        self.block6_conv2d = CustomConvBlock(Conv2D, filters=512, kernel_size=4, stride=2, padding='same',
                                             spectral_norm=True, batch_norm=False, activation=LeakyReLU,
                                             name='disc_block6')
        self.block6_global_avg = GlobalAvgPool2D(name='disc_block6_global_avg')
        self.block6_dense = SpectralNormalization(Dense(1, name='disc_block6_dense'), name='disc_block6_dense')

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
        x = self.block1_conv2d(inputs)
        x = self.block2_conv2d(x)
        x = self.block3_conv2d(x)
        x = self.block4_conv2d(x)
        x = self.block5_conv2d(x)
        x = self.block6_conv2d(x)
        x = self.block6_global_avg(x)
        x = self.block6_dense(x)
        return x

    def get_config(self):
        pass


if __name__ == '__main__':
    # gen = GeneratorNet(shape=(320, 5))
    # gen.summary()
    disc = DiscriminatorNet(input_shape=(256, 256, 3))
    disc.summary()
