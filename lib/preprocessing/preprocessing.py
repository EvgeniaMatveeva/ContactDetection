import pandas as pd
import re


def prepare_text(data: pd.DataFrame, prep_data_col: str, no_title: bool = False) -> None:
    pat_break = re.compile(r'/\n\.*')
    pat_uspace = re.compile(r'\xa0')
    pat_sep = re.compile('\*+|=+|_+|-+|\s↓+')
    pat_space = re.compile('\s+')

    data['description_prep'] = data['description'].str.replace(pat_break, ' ', regex=True)
    data['description_prep'] = data['description'].str.replace(pat_uspace, ' ', regex=True)
    data['description_prep'] = data['description'].str.replace(pat_sep, ' ', regex=True)
    data['description_prep'] = data['description'].str.replace(pat_space, ' ', regex=True)

    if no_title:
        data[prep_data_col] = data['description_prep'].values
    else:
        data[prep_data_col] = data.apply(lambda row: ' '.join((row['title'], row['description_prep'])), axis=1)
