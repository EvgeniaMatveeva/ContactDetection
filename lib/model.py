import logging
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd


class Classifier:
    def __init__(self):
        self.logger = logging.getLogger()

    def predict(self, df:  pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(df.shape[0])
        return predictions

    def predict_contact_inds(self, df: pd.DataFrame) -> Tuple[List[Optional[int]], List[Optional[int]]]:
        predictions = ([None]*df.shape[0], [None]*df.shape[0])
        return predictions
