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


# @tf.function
# def custom_l2_norm(x, epsilon=1e-12):
#     return x / (tf.reduce_sum(x ** 2) ** 0.5 + epsilon)
#
#
# @tf.function
# def power_iteration(weights, u, rounds=1):
#     _u = u
#     _v = None
#
#     for i in range(rounds):
#         _v = custom_l2_norm(tf.tensordot(_u, weights))
#         _u = custom_l2_norm(tf.tensordot(_v, tf.transpose(weights)))
#
#     weights_sn = tf.reduce_sum(tf.tensordot(_u, weights) * _v)
#     return weights_sn, _u, _v
