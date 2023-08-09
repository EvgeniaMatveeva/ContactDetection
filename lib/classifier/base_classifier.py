import re
from typing import List, Optional
import numpy as np


class ContactPattern:
    def __init__(self, contact: str, contact_labels: Optional[List[str]] = None):
        self.contact_labels = contact_labels
        self.contact = contact

    def match(self, s: str) -> bool:
        contact_matches = re.findall(self.contact, s)
        if len(contact_matches)  > 0and self.contact_labels:
            label_ptn = '|'.join(self.contact_labels)
            label_matches = re.findall(label_ptn, s, flags=re.IGNORECASE)
            return len(label_matches) > 0
        return len(contact_matches) > 0


class BaseClassifier:
    def __init__(self):
        phone = ContactPattern('\+7', ['тел', 'телефон'])
        phone_num = ContactPattern(r'\+7[^a-zA-Zа-яА-Я.,;\+]+')
        instagram = ContactPattern('@',['инстаграм', 'instagram'])
        telegram = ContactPattern('@', [' tg ', ' тг ', 'телеграм'])
        vkontakte = ContactPattern('@', ['вконтакте'])
        self.patterns = [phone, phone_num, instagram, telegram, vkontakte]

    def predict(self, text: str) -> int:
        return int(np.any([ptn.match(text) for ptn in self.patterns]))
