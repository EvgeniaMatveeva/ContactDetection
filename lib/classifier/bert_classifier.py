from typing import Tuple, List

import numpy as np
import os
from sklearn.metrics import f1_score

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from classifier.bert_dataset import BertDataset


class BertConfig:
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 32
    PRED_BATCH_SIZE = 32
    N_FOLDS = 3
    N_EPOCHS = 4
    LR = 2e-5
    BERT_FILE = 'bert_pretrained'
    TOKENIZER_FILE = 'tokenizer_pretrained'

    def __init__(self, files_path: str, model_filename: str = 'bert.pt'):
        self.bert_path = os.path.join(files_path, self.BERT_FILE)
        self.tokenizer_path = os.path.join(files_path, self.TOKENIZER_FILE)
        self.model_path = os.path.join(files_path, model_filename)


class BertClassifier:
    def __init__(self, bert_config: BertConfig, n_classes: int = 2):
        self.model = BertForSequenceClassification.from_pretrained(bert_config.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_config.tokenizer_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cfg = bert_config
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        torch.nn.init.normal_(self.model.classifier.weight, std=0.02)
        self.model.to(self.device)

        # training optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.cfg.LR, correct_bias=False)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        self.scheduler = None
        self.val_loader = None
        self.train_loader = None
        self.valid_set = None
        self.train_set = None

    def load(self, model_path: str = None):
        if model_path is None:
            model_path = self.cfg.model_path
        self.model = torch.load(model_path, map_location=torch.device(self.device))

    def save(self, model_path: str = None):
        if model_path is None:
            model_path = self.cfg.model_path
        torch.save(self.model, model_path)

    def preparation(self, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray):
        self.train_set = BertDataset(X_train, y_train, self.tokenizer, self.cfg.MAX_LEN)
        self.valid_set = BertDataset(X_valid, y_valid, self.tokenizer, self.cfg.MAX_LEN)

        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.TRAIN_BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(self.valid_set, batch_size=self.cfg.VALID_BATCH_SIZE, shuffle=True)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.cfg.N_EPOCHS
        )

    def fit(self):
        self.model.train()
        losses = []
        correct_preds = 0
        all_preds = []
        all_targets = []

        for data in self.train_loader:
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            targets = data['targets'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, targets)

            all_preds = np.hstack((all_preds, preds.cpu().detach().numpy()))
            all_targets = np.hstack((all_targets, targets.cpu().detach().numpy()))
            correct_preds += torch.sum(preds == targets)

            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        train_acc = correct_preds.double() / len(self.train_set)
        train_f1_score = f1_score(all_targets, all_preds, average='macro')
        train_loss = np.mean(losses)
        return train_acc, train_f1_score, train_loss

    def eval(self):
        self.model.eval()
        losses = []
        correct_preds = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in self.val_loader:
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                targets = data['targets'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                losses.append(loss.item())

                all_preds = np.hstack((all_preds, preds.cpu().detach().numpy()))
                all_targets = np.hstack((all_targets, targets.cpu().detach().numpy()))
                correct_preds += torch.sum(preds == targets)

        val_acc = correct_preds.double() / len(self.valid_set)
        val_f1_score = f1_score(all_targets, all_preds, average='macro')
        val_loss = np.mean(losses)
        return val_acc, val_f1_score, val_loss

    def predict(self, text: str, return_scores: bool = True) -> np.ndarray:
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.cfg.MAX_LEN,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        input_ids = out['input_ids'].to(self.device)
        attention_mask = out['attention_mask'].to(self.device)
        self.model.eval()
        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        if return_scores:
            # list of probabilities: [0-proba, 1-proba]
            prediction = torch.softmax(outputs.logits, dim=1).cpu().detach().numpy().flatten()
        else:
            prediction = torch.argmax(outputs.logits, dim=1).cpu().detach().numpy().flatten()
        return prediction

    def predict_all(self, texts: List[str]) -> np.ndarray:
        pred_set = BertDataset(texts, [],  self.tokenizer, self.cfg.MAX_LEN)
        loader = DataLoader(pred_set, batch_size=self.cfg.PRED_BATCH_SIZE, shuffle=False)
        all_preds = []

        self.model.eval()
        for data in loader:
            with torch.no_grad():
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds = np.hstack((all_preds, preds.cpu().detach().numpy()))
        return np.array(all_preds)

