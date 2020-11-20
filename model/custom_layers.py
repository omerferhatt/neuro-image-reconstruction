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
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU


class CustomConvBlock(tf.keras.layers.Layer):
    def __init__(self, conv, filters, kernel_size, stride, padding, activation,
                 spectral_norm=False, batch_norm=False, name=None):
        super(CustomConvBlock, self).__init__(name=name)
        self.sn = spectral_norm
        self.bn = batch_norm
        self.conv_type = conv
        self.activation_type = activation
        if not spectral_norm:
            self.conv = self.conv_type(filters, kernel_size, stride, padding, name=name+'_conv')
        if spectral_norm:
            self.sn = SpectralNormalization(self.conv_type(filters, kernel_size, stride, padding, name=name+'_conv'),
                                            name=name+'_conv_sn')
        if batch_norm:
            self.bn = BatchNormalization(name=name+'_bn')
        self.relu = self.activation_type(name=name + '_act')

    def call(self, inputs, *args, **kwargs):
        if not isinstance(self.sn, SpectralNormalization):
            x = self.conv(inputs)
        else:
            x = self.sn(inputs)
        if isinstance(self.bn, BatchNormalization):
            x = self.bn(x)
        return self.relu(x)


class CustomTanH(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(CustomTanH, self).__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.nn.tanh(inputs)
