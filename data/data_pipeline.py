# MIT License
#
# Copyright (c) 2020 Omer Ferhat Sarioglu
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

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# All paths will be relative to root folder of project
os.chdir('../')
DATASET_CSV_PATH = 'data/dataset.csv'


class Pipeline:
    def __init__(self, dataset_csv_path, csv_col_name='path', imagenet_col_name='inet_path', shuffle=10):
        self.dataset_csv_path = dataset_csv_path
        self.csv_col_name = csv_col_name
        self.imagenet_col_name = imagenet_col_name
        # Read EEG and image paths from .csv
        self.combined_list, self.total_record = self.read_csv()

        # Shuffle list in place
        self.shuffle = shuffle
        if shuffle > 0:
            self.random_seed = self.shuffle
            self.shuffle = True
            self.shuffle_list()
        else:
            self.shuffle = False

        # Create data generator instance
        self.gen_count = -1
        self.generator = self._create_generator()

    def read_csv(self) -> list:
        """
        Read dataset from .csv files and save paths to class
        :return: [EEG Paths, ImageNet Paths], total record count
        """
        df = pd.read_csv(self.dataset_csv_path, header=0, index_col=0)
        try:
            # Reading EEG signal and ImageNet image paths from .csv file and saves in a list
            eeg_paths = df[self.csv_col_name].values.tolist()
            inet_paths = df[self.imagenet_col_name].values.tolist()
            # Check total paths, they need to be match
            assert len(eeg_paths) == len(inet_paths), f'({len(eeg_paths)}), ({len(inet_paths)})'
            # Combine and return list
            combined_list = list(zip(eeg_paths, inet_paths))
            return [combined_list, len(combined_list)]
        except AssertionError as e:
            print(e, f'EEG file paths not matched with Imagenet file paths.')

    def shuffle_list(self, seed=None):
        """
        Shuffle list in place

        :param seed: Local seed parameter
        :return: None
        """
        # Seeding random generator to get same results in every run
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(self.random_seed)
        # Randomizing dataset in-place
        random.shuffle(self.combined_list)

    def _create_generator(self):
        """
        Creates a generator instance. It loops over all dataset.
        Every channel and image min-max normalized individually.

        :return: generator instance
        """
        while True:
            self.gen_count += 1
            if self.gen_count == self.total_record - 1:
                self.gen_count = 0
            _eeg_path, _inet_path = self.combined_list[self.gen_count]
            # Reads signals from .csv and converting to NumPy array
            _eeg_sig = pd.read_csv(_eeg_path, header=None, index_col=0).values
            # Removing first 32 sample and reading total (32*10) sample from signals
            # This process needed for removing some noise from data
            _eeg_sig = _eeg_sig[:, 32:352]
            # Min-Max normalization within channel
            _eeg_sig = np.asarray(_eeg_sig, dtype=np.float32)
            _eeg_sig = _eeg_sig.T
            _eeg_sig = _eeg_sig[np.newaxis, :, :]
            # Reading and resizing image to (224, 224). This resolution is compatible with most of state of art model
            # e.g. ResNet, VGGNet etc.
            _img = Image.open(_inet_path).resize((256, 256))
            if _img.mode != 'RGB':
                _img = _img.convert('RGB')
            # Creating batch channel
            _img = np.asarray(_img, dtype=np.float32)[np.newaxis, :, :, :]
            # Min-Max normalization in image
            _img = (_img - 127.5) / 127.5
            _img_sig = [_eeg_sig, _img]
            yield _img_sig

    def _stop_generator(self):
        self.generator.close()


def visualize(signal, img, signal_channel=0, show_results=True, save_path=None):
    """Visualize image and related EEG signal with one channel"""
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(12, 5), dpi=300)
    axs[0].imshow(img[0, :, :, :])
    axs[1].plot(signal[signal_channel, :])
    if show_results:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)


if __name__ == '__main__':
    # Create a Pipeline instance
    pipeline = Pipeline(DATASET_CSV_PATH, shuffle=10)
    # Get normalize EEG signal and image
    eeg_signal, image = next(pipeline.generator)
    # Visualize data and save result
    visualize(eeg_signal, image, show_results=True, save_path='data/visualize_result/vis.png')
