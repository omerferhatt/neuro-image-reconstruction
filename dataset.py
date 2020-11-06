import os
import glob
from tqdm.auto import tqdm
import pandas as pd
from doltpy.core import Dolt, ServerConfig
from doltpy.core.read import read_table, pandas_read_sql


DATA_DIR_PATH = 'mind-big-data/data'
DATASET_CSV_PATH = 'mind-big-data/dataset.csv'
WORDNET_PATH = 'data/WordReport-v1.04.txt'
IMAGENET_SQL_PATH = 'image-net'


class ImagenetSQL:
    def __init__(self, repo_dir, remote_url=''):
        self.repo_dir = repo_dir
        self.remote_url = remote_url
        self.repo = self.__init_repo()
        self.repo.sql_server()

    def __init_repo(self):
        try:
            if os.path.exists(self.repo_dir):
                dolt_sv_conf = ServerConfig()
                repo = Dolt(self.repo_dir, server_config=dolt_sv_conf)
                return repo
            else:
                repo = Dolt.clone(self.remote_url)
                return repo
        except ValueError:
            print('Dolt repo init error!')

    def read_query_table(self, query):
        return pandas_read_sql(query, self.repo.get_engine())

    def read_table(self):
        df = read_table(self.repo, 'images_synsets')
        return df

    def stop_sql(self):
        self.repo.sql_server_stop()


class DatasetWNParser:
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
            'image_url': []
        }
        self.filenames = self.get_file_names()

    def get_file_names(self):
        return glob.glob(os.path.join(self.path, '*.csv'))

    def parse(self, sql_engine: ImagenetSQL):
        for file_name in tqdm(self.filenames):
            splitted_list = file_name.split(sep='/t')[-1].split(sep='_')
            self.data_dict['path'].append(file_name)
            self.data_dict['dataset'].append(splitted_list[1])
            self.data_dict['device'].append(splitted_list[2])
            self.data_dict['wn_id'].append(splitted_list[3][1:])
            self.data_dict['im_id'].append(splitted_list[4])
            self.data_dict['eeg_session'].append(splitted_list[5])
            self.data_dict['global_session'].append(splitted_list[6].split(sep='.')[0])
            query = f"SELECT image_url FROM images_synsets WHERE synset_id='{splitted_list[3][1:]}' AND image_id={splitted_list[4]};"
            df = sql_engine.read_query_table(query=query)
            if df.empty:
                self.data_dict['image_url'].append('')
            else:
                self.data_dict['image_url'].append(df['image_url'][0])
        sql_engine.stop_sql()

    def save_dataset_csv(self, path):
        cols = list(self.data_dict.keys())
        df = pd.DataFrame(self.data_dict, index=None, columns=cols)
        df.to_csv(path, index=True)


if __name__ == '__main__':
    # if not os.path.exists(DATASET_CSV_PATH):
    inet_sql = ImagenetSQL(IMAGENET_SQL_PATH, 'dolthub/image-net')
    wn_parser = DatasetWNParser(DATA_DIR_PATH)
    wn_parser.parse(sql_engine=inet_sql)
    wn_parser.save_dataset_csv(path=DATASET_CSV_PATH)


