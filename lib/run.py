import argparse
import logging
import os
import subprocess
import sys
from typing import Tuple

import pandas as pd
from model import Classifier


class Test:
    train_csv = 'train.csv'
    val_csv = 'val.csv'
    test_csv = 'test.csv'
    contacts_prediction = 'contacts_prediction.csv'
    contacts_segmentation = 'contacts_segmentation.csv'

    def __init__(self, debug: bool = True):
        self.debug = debug
        self.logger = self._get_logger()
        self.data_dir = os.getenv('DATA_ROOT')
        self.output_dir = os.getenv('OUTPUT_ROOT')
        self.user = os.getenv('USER')

    def _get_logger(self):
        logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def train(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, self.train_csv)
        if not os.path.exists(path):
            self.logger.info(f'File {path} not found')
            raise ValueError('train not found')
        df = pd.read_csv(path)

        return df

    def test(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, self.test_csv)
        if not os.path.exists(path):
            self.logger.info(f'File {path} not found')
            raise ValueError('test not found')
        df = pd.read_csv(path)
        if 'is_bad' in df:
            del df['is_bad']

        return df

    def run(self):
        self.logger.info(f'Starting model on behalf of {self.user}')
        contacts_prediction, contacts_segmentation = self.process()
        contacts_prediction.index = contacts_prediction['index']
        contacts_segmentation.index = contacts_segmentation['index']

        prediction_path = os.path.join(self.output_dir, f'{self.user}_{self.contacts_prediction}')
        self.logger.info(f'Saving results {prediction_path}')
        contacts_prediction.to_csv(prediction_path, index=False)

        segmentation_path = os.path.join(self.output_dir, f'{self.user}_{self.contacts_segmentation}')
        self.logger.info(f'Saving results {segmentation_path}')
        contacts_segmentation.to_csv(segmentation_path, index=False)

        self.logger.info('Done')

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_df = self.test()

        contacts_prediction = pd.DataFrame(columns=['index', 'prediction'])
        contacts_prediction['index'] = test_df.index

        model = Classifier()
        prediction = model.predict(test_df)
        contacts_prediction['prediction'] = prediction

        contacts_segmentation = pd.DataFrame(columns=['index', 'start', 'finish'])
        contacts_segmentation['index'] = test_df.index
        start, finish = model.predict_contact_inds(test_df)
        contacts_segmentation['start'] = start
        contacts_segmentation['finish'] = finish

        return contacts_prediction, contacts_prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')

    # setting USER variable as it is not initialized in container
    whoami = subprocess.check_output('whoami', shell=True).strip().decode('utf-8')
    os.environ['USER'] = whoami

    test = Test(debug=parser.parse_args().debug)
    test.run()
