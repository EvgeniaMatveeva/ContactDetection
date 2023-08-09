import logging
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from classifier.base_classifier import BaseClassifier
from preprocessing.preprocessing import prepare_text


class Classifier:
    TEXT_COL = 'text'

    def __init__(self):
        self.logger = logging.getLogger()

    def predict(self, df:  pd.DataFrame) -> np.ndarray:
        df.reset_index(inplace=True)
        prepare_text(df, prep_data_col=self.TEXT_COL)

        classifier = BaseClassifier()
        texts = df[self.TEXT_COL].values
        print(len(texts))
        predictions = [classifier.predict(t) for t in texts]
        return predictions

    def predict_contact_inds(self, df: pd.DataFrame) -> Tuple[List[Optional[int]], List[Optional[int]]]:
        predictions = ([None]*df.shape[0], [None]*df.shape[0])
        return predictions
