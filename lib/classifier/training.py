import logging

import numpy as np
import time

import pandas as pd

from classifier.bert_classifier import BertClassifier


logger = logging.getLogger()


def train_val_split(data: pd.DataFrame, n_folds: int):
    folds = range(n_folds - 1)
    fold_size = data.shape[0] // n_folds
    logger.info(f'Train size={data.shape[0]}, fold size={fold_size}')
    free_inds = set(range(data.shape[0]))
    for i in folds:
        fold_inds = np.random.choice(list(free_inds), fold_size, replace=False)
        if len(fold_inds) > 0:
            data.loc[fold_inds, 'fold'] = i
        free_inds -= set(fold_inds)
    fold_inds = list(free_inds)
    data.loc[fold_inds, 'fold'] = n_folds - 1
    data['fold'] = data['fold'].astype(int)


def train_classifier(classifier: BertClassifier, train_data:  pd.DataFrame, data_col: str, label_col: str,
                     n_epochs: int = 1, n_folds: int = 3):
    train_val_split(train_data, n_folds=n_folds)

    start_time = time.time()
    best_score = 0
    for epoch in range(n_epochs):
        logger.info(f'Epoch {epoch + 1}/{n_epochs}')
        train_acc, train_score, train_loss = [], [], []
        val_acc, val_score, val_loss = [], [], []
        for val_fold in range(n_folds):
            logger.info(f'Fold {val_fold}')
            val_fold_data = train_data[train_data['fold'] == val_fold]
            train_fold_data = train_data[train_data['fold'] != val_fold]
            classifier.preparation(
                X_train=train_fold_data[data_col].values,
                y_train=train_fold_data[label_col].values,
                X_valid=val_fold_data[data_col].values,
                y_valid=val_fold_data[label_col].values
            )

            train_fold_acc, train_fold_score, train_fold_loss = classifier.fit()
            train_acc.append(train_fold_acc.cpu())
            train_score.append(train_fold_score)
            train_loss.append(train_fold_loss)

            val_fold_acc, val_fold_score, val_fold_loss = classifier.eval()
            val_acc.append(val_fold_acc.cpu())
            val_score.append(val_fold_score)
            val_loss.append(val_fold_loss)
            logger.info('-' * 10)
        mean_train_loss = np.mean(train_loss)
        mean_train_acc = np.mean(train_acc)
        mean_train_score = np.mean(train_score)
        logger.info(f'Train loss {mean_train_loss:10.4f}, accuracy {mean_train_acc:10.4f}, f1-score {mean_train_score:10.4f}')
        mean_val_loss = np.mean(val_loss)
        mean_val_acc = np.mean(val_acc)
        mean_val_score = np.mean(val_score)
        logger.info(f'Val loss   {mean_val_loss:10.4f}, accuracy {mean_val_acc:10.4f}, f1-score {mean_val_score:10.4f}')

        if mean_val_acc > best_score:
            logger.info(f'Saving model to {classifier.cfg.model_path}')
            classifier.save()
            best_score = mean_val_acc

    classifier.load()

    logger.info('-' * 10)
    logger.info(f'Training time:   --- {(time.time() - start_time) / 60:f} minutes ---')