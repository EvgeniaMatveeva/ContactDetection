import logging
import os
import pickle
import time
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from classifier.bert_classifier import BertClassifier, BertConfig
from classifier.training import train_classifier
from classifier.selecting import split_words, get_contact_inds
from preprocessing.preprocessing import prepare_text


class Classifier:
    MAX_TRAIN_SIZE = 200000
    N_CLASSES = 2
    TEXT_COL = 'text'
    LABEL_COL = 'is_bad'
    CAT_COL = 'category'
    CAT2IND_FILE = 'cat2ind.pkl'
    BERTCAT_FILE = 'bert_cat%i.pt'

    # parameters for searching for contacts in text
    WINDOW_SIZE = 3
    OVERLAP_SIZE = 2
    MAX_WORD_COUNT4SELECT = 10
    BERT4CONTACT_SELECT_FILE = 'bert.pt'
    DESCRIPTION_COL = 'description'

    def __init__(self, files_path: str, load_model: bool = False):
        self.logger = logging.getLogger()
        self.files_path = files_path
        self.bert_cfg = BertConfig(files_path)
        self.default_model = None
        self.model4select = None
        self.catind2model = None
        self.cat2ind = None
        self.ind2cat = None
        if load_model:
            self._load_models()

    def _load_models(self):
        with open(os.path.join(self.files_path, self.CAT2IND_FILE), 'rb') as f:
            self.cat2ind = pickle.load(f)
        self.ind2cat = {i: cat for cat, i in self.cat2ind.items()}
        self.catind2model = {}
        for cat, cat_num in self.cat2ind.items():
            cfg = BertConfig(self.files_path, self.BERTCAT_FILE % cat_num)
            bert_cat = BertClassifier(bert_config=cfg, n_classes=self.N_CLASSES)
            bert_cat.load()
            self.catind2model[cat_num] = bert_cat

        self.default_model = BertClassifier(bert_config=self.bert_cfg, n_classes=self.N_CLASSES)
        self.default_model.load()

    def predict(self, df:  pd.DataFrame) -> List[float]:
        if self.default_model is None:
            self.logger.warning('Model is not trained!')
            return

        start_time = time.time()
        self.logger.info(f'Preprocessing text...')
        df.reset_index(inplace=True)
        prepare_text(df, prep_data_col=self.TEXT_COL)
        predictions = np.zeros(df.shape[0])

        cat_num_col = self.CAT_COL+'_i'
        default_cat_num = -1
        df[cat_num_col] = df[self.CAT_COL].map(self.cat2ind).fillna(default_cat_num).astype(int)
        cat_inds = list(self.cat2ind.values()) + [default_cat_num]

        self.logger.info(f'Generating contact predictions...')
        for cat_num in cat_inds:

            df_cat_inds = df[df[cat_num_col] == cat_num].index.values
            classifier = self.catind2model.get(cat_num, self.default_model)
            texts = df[self.TEXT_COL].values[df_cat_inds]
            cat_predictions = [classifier.predict(t) for t in texts]
            cat_proba_predictions = [p1 for p0, p1 in cat_predictions]
            predictions[df_cat_inds] = cat_proba_predictions

        self.logger.info(f'Time generating predictions:   --- {(time.time() - start_time) / 60:f} minutes ---')
        return predictions

    def train(self, train_data: pd.DataFrame):
        if train_data.shape[0] > self.MAX_TRAIN_SIZE:
            train_data = self._sample_data(train_data, self.MAX_TRAIN_SIZE)

        start_time = time.time()
        prepare_text(train_data, prep_data_col=self.TEXT_COL)

        categories = list(set(train_data[self.CAT_COL].values))
        self.cat2ind = {cat: i for i, cat in enumerate(categories)}
        cat_num_col = self.CAT_COL+'_i'
        train_data[cat_num_col] = train_data[self.CAT_COL].map(self.cat2ind)

        self.catind2model = {}
        for cat, cat_num in self.cat2ind.items():
            self.logger.info(f'Training model for category {cat}...')

            train_cat_data = train_data[train_data[cat_num_col] == cat_num]
            train_cat_data.reset_index(inplace=True)
            self.catind2model[cat_num] = self._train_cat_bert(train_cat_data, cat_num)

            self.logger.info(f'Time training:   --- {(time.time() - start_time) / 60:f} minutes ---')

        self.default_model = BertClassifier(self.bert_cfg, n_classes=self.N_CLASSES)
        train_classifier(self.default_model,
                         train_data,
                         data_col=self.TEXT_COL,
                         label_col=self.LABEL_COL,
                         n_epochs=self.bert_cfg.N_EPOCHS,
                         n_folds=self.bert_cfg.N_FOLDS)
        self.logger.info(f'Total time training:   --- {(time.time() - start_time) / 60:f} minutes ---')

    def _train_cat_bert(self, train_data: pd.DataFrame, cat_num: int) -> BertClassifier:
        cfg = BertConfig(self.files_path, self.BERTCAT_FILE % cat_num)
        self.logger.info(f'Sample size: {train_data.shape[0]}')

        bert_cat = BertClassifier(bert_config=cfg, n_classes=self.N_CLASSES)
        train_classifier(bert_cat,
                         train_data,
                         data_col=self.TEXT_COL,
                         label_col=self.LABEL_COL,
                         n_epochs=self.bert_cfg.N_EPOCHS,
                         n_folds=self.bert_cfg.N_FOLDS)
        return bert_cat

    def _sample_data(self, train_data: pd.DataFrame, sample_size=MAX_TRAIN_SIZE) -> pd.DataFrame:
        self.logger.info(f'All data size: {train_data.shape}')
        inds = np.random.choice(list(range(train_data.shape[0])), sample_size, replace=False)
        train = train_data.loc[inds]
        train.reset_index(inplace=True)
        self.logger.info(f'Selected data size: {train.shape}')
        return train

    def predict_contact_inds(self, df: pd.DataFrame, has_contact_preds: List[int] = None) -> Tuple[List[Optional[int]], List[Optional[int]]]:
        cfg = BertConfig(self.files_path, self.BERT4CONTACT_SELECT_FILE)
        cfg.MAX_LEN = self.WINDOW_SIZE
        self.model4select = BertClassifier(bert_config=cfg)
        self.model4select.load()
        self.logger.info(f'Model is loaded')

        start_time = time.time()
        self.logger.info(f'Preprocessing text...')
        df.reset_index(inplace=True)
        prepare_text(df, prep_data_col=self.DESCRIPTION_COL, no_title=True)

        self.logger.info(f'Searching for contacts in texts...')
        texts = df[self.DESCRIPTION_COL].values
        N = len(texts)
        start_inds = [None] * N
        end_inds = [None] * N

        has_contact_inds = [i for i, v in enumerate(has_contact_preds) if v == 1]
        # search for contacts indices in texts, which have high probability of contact info present
        for ind in tqdm(has_contact_inds):
            text = texts[ind]
            words = str.split(text)
            min_index = 0
            if len(words) > self.MAX_WORD_COUNT4SELECT:
                left, words_subset = self._split_check(words, 0, len(words) - 1, self.MAX_WORD_COUNT4SELECT)
                min_index = sum([len(w) + 1 for w in words[0: left]])
                if not words_subset:
                    continue
            else:
                words_subset = words

            is_contact = self._predict_contact_words(words_subset)
            start, end = get_contact_inds(text, words_subset, min_index, is_contact)
            if end is not None:
                self.logger.info(f'Contact found in text[{start}:{end}]: {text[start: end + 1]}')
                start_inds[ind] = start
                end_inds[ind] = end

        self.logger.info(f'Total time searching for contacts:   --- {(time.time() - start_time) / 60:f} minutes ---')
        return start_inds, end_inds

    def _split_check(self, words: List[str], left: int, right: int, max_len: int = 20) -> Tuple[int, List[str]]:
        med = left + (right - left) // 2
        has_contact_left = self.default_model.predict(' '.join(words[left: med]), return_scores=False)[0]
        has_contact_right = self.default_model.predict(' '.join(words[med: right + 1]), return_scores=False)[0]
        if has_contact_left == has_contact_right or right - left <= max_len:
            if has_contact_left == 1 or right - left <= max_len:
                return left, words[left: right + 1]
            else:
                return 0, []

        if has_contact_left == 1:
            return self._split_check(words, left, med, max_len)
        if has_contact_right == 1:
            return self._split_check(words, med, right, max_len)

    def _predict_contact_words(self, words: List[str]) -> Dict[str, int]:
        split_texts = split_words(words, self.WINDOW_SIZE, self.OVERLAP_SIZE)
        all_preds = self.model4select.predict_all(split_texts)
        is_contact = {}
        for i, word in enumerate(words):
            if i == 0:
                is_contact[word] = int(all_preds[i])
            elif i == 1:
                is_contact[word] = int(max(all_preds[[0, 1]]))
            else:
                is_contact[word] = int(max(all_preds[i - self.WINDOW_SIZE + 1: i + 1]))
        return is_contact