import os
import glob
import argparse
import pandas as pd
from tqdm.auto import tqdm

# All paths will be relative to root folder of project
os.chdir('..')


class DatasetWNParser:
    """
    Maps local file names with original ImageNet 2013 training files

    :param path: Path to data directory
    """
    def __init__(self, path):
        self.path = path
        self.data_dict = {
            'path': [],
            'dataset': [],
            'device': [],
            'wn_id': [],
            'im_id': [],
            'eeg_session': [],
            'global_session': [],
            'inet_path': []
        }
        self.filenames = self.get_file_names()

    def get_file_names(self):
        """Reads files names from data directory as a list"""
        return glob.glob(os.path.join(self.path, '*.csv'))

    def parse_and_map(self, local_inet_path):
        """
        Saves file information with mapped ImageNet path to a dictionary

        :param local_inet_path: Path to ImageNet folder
        """
        for file_name in tqdm(self.filenames):
            # Reads file name from full file path
            sliced_list = file_name.split(sep='/t')[-1].split(sep='_')
            self.data_dict['path'].append(file_name)
            self.data_dict['dataset'].append(sliced_list[1])
            self.data_dict['device'].append(sliced_list[2])
            self.data_dict['wn_id'].append(sliced_list[3])
            self.data_dict['im_id'].append(sliced_list[4])
            self.data_dict['eeg_session'].append(sliced_list[5])
            self.data_dict['global_session'].append(sliced_list[6].split(sep='.')[0])
            # File name: /MindBigData_Imagenet_Insight_n00007846_6247_1_785
            # Imagenet file path: /n00007846/n00007846_6247.JPEG
            file_name = str(sliced_list[3] + '_' + sliced_list[4] + '.JPEG')
            inet_path = os.path.join(local_inet_path, sliced_list[3], file_name)
            self.data_dict['inet_path'].append(inet_path)

    def save_dataset_csv(self, path):
        """
        Saves dictionary file to disk with all information

        :param path: Save path
        :return: Saved pandas.DataFrame into disk as .csv file
        """
        cols = list(self.data_dict.keys())
        df = pd.DataFrame(self.data_dict, index=None, columns=cols)
        df.to_csv(path, index=True)


def main(arguments):
    # Creates a data parser instance
    wn_parser = DatasetWNParser(arguments.data_dir_path)
    # Maps local file names with ImageNet image paths
    wn_parser.parse_and_map(local_inet_path=arguments.local_inet_path)
    # Saves mapped data as a .csv
    wn_parser.save_dataset_csv(path=arguments.csv_save_path)


if __name__ == '__main__':
    # Arguments for general usage
    parser = argparse.ArgumentParser(description='Data parser')
    parser.add_argument('--data-dir-path', type=str, default='data',
                        help='Path to directory which includes all .csv files in it')
    parser.add_argument('--local-inet-path', type=str, required=True,
                        help='Path to local ImageNet 2013 Training folder')
    parser.add_argument('--csv-save-path', type=str, default='dataset.csv',
                        help='Output save path')
    arg = parser.parse_args()
    # Program
    main(arg)
