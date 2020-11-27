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
from tensorflow.keras.layers import Add, Reshape, Concatenate
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Input, Conv1D, Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import MaxPool1D, GlobalMaxPool1D
from tensorflow.keras.models import Model


class FeatureExtractor:
    def __init__(self, input_shape=(320, 5)):
        self.input_shape = input_shape
        self.model = self.get_model()

    def get_model(self):
        inp = Input(shape=self.input_shape)

        x_b1_out1 = Conv1D(128, kernel_size=13, strides=1, padding='same', activation='relu')(inp)
        x_b1_out1 = BatchNormalization()(x_b1_out1)
        x_b1_out2 = Conv1D(128, kernel_size=9, strides=1, padding='same', activation='relu')(inp)
        x_b1_out2 = BatchNormalization()(x_b1_out2)
        x_b1_out = Concatenate(axis=-1)([x_b1_out1, x_b1_out2])
        x = MaxPool1D()(x_b1_out)

        x_b2_out1 = Conv1D(128, kernel_size=9, strides=1, padding='same', activation='relu')(x)
        x_b2_out1 = BatchNormalization()(x_b2_out1)
        x_b2_out2 = Conv1D(128, kernel_size=7, strides=1, padding='same', activation='relu')(x)
        x_b2_out2 = BatchNormalization()(x_b2_out2)
        x_b2_out3 = Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu')(x)
        x_b2_out3 = BatchNormalization()(x_b2_out3)
        x_b2_out = Concatenate(axis=-1)([x_b2_out1, x_b2_out2, x_b2_out3])
        x = MaxPool1D()(x_b2_out)

        x_b3_out1 = Conv1D(64, kernel_size=7, strides=1, padding='same', activation='relu')(x)
        x_b3_out1 = BatchNormalization()(x_b3_out1)
        x_b3_out2 = Conv1D(64, kernel_size=7, strides=1, padding='same', activation='relu')(x_b3_out1)
        x_b3_out2 = BatchNormalization()(x_b3_out2)
        x_b3_out = Add()([x_b3_out1, x_b3_out2])
        x = MaxPool1D()(x_b3_out)

        x_b4_out1 = Conv1D(32, kernel_size=7, strides=1, padding='same', activation='relu')(x)
        x_b4_out1 = BatchNormalization()(x_b4_out1)
        x_b4_out2 = Conv1D(32, kernel_size=7, strides=1, padding='same', activation='relu')(x_b4_out1)
        x_b4_out2 = BatchNormalization()(x_b4_out2)
        x_b4_out = Add()([x_b4_out1, x_b4_out2])
        x = MaxPool1D()(x_b4_out)

        x_b5_out1 = Conv1D(32, kernel_size=5, strides=1, padding='same', activation='relu')(x)
        x_b5_out1 = BatchNormalization()(x_b5_out1)
        x_b5_out2 = Conv1D(32, kernel_size=5, strides=1, padding='same', activation='relu')(x_b5_out1)
        x_b5_out2 = BatchNormalization()(x_b5_out2)
        x_b5_out = Add()([x_b5_out1, x_b5_out2])
        x = GlobalMaxPool1D()(x_b5_out)

        x = Dense(32, activation='tanh')(x)
        x = Dropout(0.1)(x)
        out = Dense(16, activation='sigmoid')(x)

        model = Model(inputs=inp, outputs=out)
        return model


class Generator:
    def __init__(self, input_cond=(16,), input_rand=(100,)):
        self.input_cond = input_cond
        self.input_rand = input_rand
        self.model = self.get_model()

    def get_model(self):
        inp_cond = Input(shape=self.input_cond)
        inp_rand = Input(shape=self.input_rand)
        inp_concat = Concatenate()([inp_cond, inp_rand])
        x = Dense(512, activation='relu')(inp_concat)
        x = Reshape(target_shape=(4, 4, 32))(x)
        x = Conv2DTranspose(256, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(256, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(3, (7, 7), activation='tanh', padding='same')(x)

        model = Model(inputs=[inp_rand, inp_cond], outputs=x)
        return model


if __name__ == '__main__':
    # gen = GeneratorNet(shape=(320, 5))
    # gen.summary()
    # disc = DiscriminatorNet(shape=(256, 256, 3))
    # disc.summary()
    feature_extractor = FeatureExtractor(input_shape=(320, 5))
    feature_extractor.model.summary()
    tf.keras.utils.plot_model(
        feature_extractor.model,
        to_file="model_ext.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

    generator = Generator()
    generator.model.summary()
    tf.keras.utils.plot_model(
        generator.model,
        to_file="model_gen.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )
