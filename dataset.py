import os
import pandas as pd
from doltpy.core import Dolt, ServerConfig
from doltpy.core.read import read_table, pandas_read_sql


DATA_DIR_PATH = 'data/MindBigData-Imagenet'
WORDNET_PATH = 'data/WordReport-v1.04.txt'
IMAGENET_SQL_PATH = 'image-net'


def read_wordnet(wordnet_path):
    """

    :param wordnet_path:
    :return:
    """
    synset_id = []
    image_count = []
    labels = []
    with open(wordnet_path) as file:
        data = file.readlines()
        for row in data:
            row = row.split(sep='\t')
            labels.append(row[0].split(sep=', '))
            image_count.append(row[1])
            synset_id.append(row[2])
    return synset_id, image_count, labels


def create_dataset_csv(data_dir_path):
    """

    :param data_dir_path:
    :return:
    """
    fp = sorted(os.listdir(data_dir_path))
    dataset_dict = {
        'dataset_source': [],
        'image_source': [],
        'device': [],
        'wnid': [],

    }


class ImagenetSQL:
    def __init__(self, repo_dir, remote_url=''):
        self.repo_dir = repo_dir
        self.remote_url = remote_url
        self.repo = self.__init_repo()

    def __init_repo(self):
        try:
            if os.path.exists(self.repo_dir):
                dolt_sv_conf = ServerConfig()
                repo = Dolt(self.repo_dir, server_config=dolt_sv_conf)
                return repo
            else:
                repo = Dolt.clone(self.remote_url)
                return repo
        except:
            print('Dolt repo error!')

    def read_query_table(self, query):
        self.repo.sql_server()
        df = pandas_read_sql(query, self.repo.get_engine())
        self.repo.sql_server_stop()
        return df

    def read_table(self):
        df = read_table(self.repo, 'images_synsets')
        return df


if __name__ == '__main__':
    # wordnet_id, im_count, labels = read_wordnet(WORDNET_PATH)
    # create_dataset_csv(DATA_DIR_PATH)
    imnet_sql = ImagenetSQL(IMAGENET_SQL_PATH, 'dolthub/image-net')
