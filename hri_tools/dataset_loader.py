import os
import json
import re
from collections import Counter
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

from .utils import SUPPORTED_DATASETS, DATA_PATH

class AbstractHumorDataset(ABC):
    
    '''
    # TODO add doc strings
    the class
    '''
    
    def __init__(self, name) -> None:
        if name not in SUPPORTED_DATASETS:
            raise ValueError('This name are not supported')
        self.name = name
        self.full_name = None
        self.descriprion = None
        self.df = None
        self.authors = None
        self.first_mention = None
        self.other_mention = None
        self.year = None
        self._dataset_preprocessed = False
        self._vocab_built = False

    @abstractmethod
    def load(self):
        pass

    def _calc_mean_length_by_word(self):
        statistics = list()
        texts = self.df['text'].tolist()
        for i in range(len(texts)):
            statistics.append(
                len(
                    word_tokenize(str(texts[i]))
                )
            )
        
        self.mean_word_length = np.mean(statistics)

        statistics_pos = list()
        texts = self.df[['text', 'label']]
        texts = texts[texts["label"] == 1]
        texts = texts["text"].tolist()
        for i in range(len(texts)):
            statistics_pos.append(
                len(
                    word_tokenize(str(texts[i]))
                )
            )
        
        self.mean_word_length_pos = np.mean(statistics_pos)

        statistics_neg = list()
        texts = self.df[['text', 'label']]
        texts = texts[texts["label"] == 0]
        texts = texts["text"].tolist()
        for i in range(len(texts)):
            statistics_neg.append(
                len(
                    word_tokenize(str(texts[i]))
                )
            )
        
        self.mean_word_length_neg = np.mean(statistics_neg)

    def _calc_mean_length_by_symbols_with_space(self):
        statistics = list()
        texts = self.df['text'].tolist()
        for i in range(len(texts)):
            statistics.append(
                len(
                    texts[i]
                )
            )
        
        self.mean_length_by_symbols_with_space = np.mean(statistics)

    def _calc_mean_length_by_symbols_without_space(self):
        statistics = list()
        texts = self.df['text'].tolist()
        for i in range(len(texts)):
            statistics.append(
                len(
                    str(texts[i]).replace(' ', '')
                )
            )
        
        self.mean_length_by_symbols_without_space = np.mean(statistics)

    def _calc_class_balance(self):
        self.number_of_records = len(self.df)
        self.pos_num_rec = len(self.df[self.df['label'] == 1])
        self.neg_num_rec = len(self.df[self.df['label'] == 0])

        # train part
        self.number_of_records_train = len(self.train_df)
        self.pos_num_rec_train = len(self.train_df[self.train_df['label'] == 1])
        self.neg_num_rec_train = len(self.train_df[self.train_df['label'] == 0])

        # test part
        self.number_of_records_test = len(self.test_df)
        self.pos_num_rec_test = len(self.test_df[self.test_df['label'] == 1])
        self.neg_num_rec_test = len(self.test_df[self.test_df['label'] == 0])

        # dev part
        self.number_of_records_valid = len(self.valid_df)
        self.pos_num_rec_valid = len(self.valid_df[self.valid_df['label'] == 1])
        self.neg_num_rec_valid = len(self.valid_df[self.valid_df['label'] == 0])

    def calc_statistics(self):
        self._calc_mean_length_by_word()
        self._calc_mean_length_by_symbols_with_space()
        self._calc_mean_length_by_symbols_without_space()
        self._calc_class_balance()

    def print_statistics_report(self):
        text = f'''
        Statistics report for {self.name} dataset:
        Number of samples: {self.number_of_records}, pos - {self.pos_num_rec}, neg - {self.neg_num_rec}.

        Stats for train/test/valid:
        Number of samples train: {self.number_of_records_train}, pos - {self.pos_num_rec_train}, neg - {self.neg_num_rec_train}.
        Number of samples test: {self.number_of_records_test}, pos - {self.pos_num_rec_test}, neg - {self.neg_num_rec_test}.
        Number of samples valid: {self.number_of_records_valid}, pos - {self.pos_num_rec_valid}, neg - {self.neg_num_rec_valid}.

        Mean word length is {self.mean_word_length}, pos - {self.mean_word_length_pos}, neg - {self.mean_word_length_neg}
        Mean length by symbols is {self.mean_length_by_symbols_with_space} (without space is {self.mean_length_by_symbols_without_space})
        '''
        print(text)

    def get_positive_sample(self):
        tmp_df = self.df[self.df['label'] == 1]
        return tmp_df.sample(1)['text'].iloc[0]

    def print_positive_sample(self):
        sample = self.get_positive_sample()
        text = f'''
        Example from "{self.name}" Dataset
        Humorous eaxmple: {sample}
        '''
        print(text)

    def __str__(self):
        about = f'''
            The Dataset name is {self.full_name}
            The authors is {self.authors}
            year of publishing is {self.year}
            Short descriprion: {self.descriprion}
            Paper with first mentions: {self.first_mention}
            Other mentions paper: {self.other_mention}
        '''
        return about
    
    def _to_lower_case_rows(self):
        self.df['text_preprocessed'] = self.df['text'].apply(
            lambda row: str(row).lower()
        )

        self.train_df['text_preprocessed'] = self.train_df['text'].apply(
            lambda row: str(row).lower()
        )

        self.test_df['text_preprocessed'] = self.test_df['text'].apply(
            lambda row: str(row).lower()
        )

        self.valid_df['text_preprocessed'] = self.valid_df['text'].apply(
            lambda row: str(row).lower()
        ) 

    def _remove_symbols_from_rows(self):
        # saving only letters and digits
        self.df['text_preprocessed'] = self.df['text_preprocessed'].apply(
            lambda row: re.sub(pattern='[^\w ]', repl='', string=row)
        )

        self.train_df['text_preprocessed'] = self.train_df['text_preprocessed'].apply(
            lambda row: re.sub(pattern='[^\w ]', repl='', string=row)
        )

        self.test_df['text_preprocessed'] = self.test_df['text_preprocessed'].apply(
            lambda row: re.sub(pattern='[^\w ]', repl='', string=row)
        )

        self.valid_df['text_preprocessed'] = self.valid_df['text_preprocessed'].apply(
            lambda row: re.sub(pattern='[^\w ]', repl='', string=row)
        )
    
    def _word_tokenize_rows(self):
        self.df['text_preprocessed'] = self.df['text_preprocessed'].apply(
            lambda row: word_tokenize(row)
        )

        self.train_df['text_preprocessed'] = self.train_df['text_preprocessed'].apply(
            lambda row: word_tokenize(row)
        )

        self.test_df['text_preprocessed'] = self.test_df['text_preprocessed'].apply(
            lambda row: word_tokenize(row)
        )

        self.valid_df['text_preprocessed'] = self.valid_df['text_preprocessed'].apply(
            lambda row: word_tokenize(row)
        )

    def run_preprocessing(self):
        self._to_lower_case_rows()
        self._remove_symbols_from_rows()
        self._word_tokenize_rows()
        self._dataset_preprocessed = True

    def build_vocab(self):

        if not(self.is_preprocessed()):
            raise ValueError('You shoud preprocessing data')

        all_words = list()
        for text in self.df['text_preprocessed']:
            all_words += text

        self.vocab = dict(Counter(all_words))
        self.vocab_size = len(self.vocab)
        self.non_unique_words = sum([w[1] for w in self.vocab.items()])
        self._vocab_built = True
    
    def is_preprocessed(self):
        return self._dataset_preprocessed
    
    def is_vocab_built(self):
        return self._vocab_built

    def get_train(self):
        return self.train_df

    def get_test(self):
        return self.test_df

    def get_valid(self):
        return self.valid_df
    

class HumorDataset(AbstractHumorDataset):
    """ #TODO FIX DOCS """

    def __init__(self, name) -> None:
        super().__init__(name)
        if name == "meta_dataset":
            raise ValueError("meta_dataset unsupported by this class")

    def _load_dataframe(self):

        # Not use this data frame. Because in this df there is dubs
        # self.df = pd.read_csv(
        #     os.path.join(os.getenv('HOME'), DATA_PATH, 'datasets', f'{self.name}', 'files', 'data.csv'),
        #     index_col=0
        # )

        self.train_df = pd.read_csv(
            os.path.join(os.getenv('HOME'), DATA_PATH, 'datasets', f'{self.name}', 'files', 'train.csv'),
            index_col=0
        )

        self.valid_df = pd.read_csv(
            os.path.join(os.getenv('HOME'), DATA_PATH, 'datasets', f'{self.name}', 'files', 'valid.csv'),
            index_col=0
        )

        self.test_df = pd.read_csv(
            os.path.join(os.getenv('HOME'), DATA_PATH, 'datasets', f'{self.name}', 'files', 'test.csv'),
            index_col=0
        )

        self.df = pd.concat([self.train_df.copy(), self.test_df.copy(), self.valid_df.copy()])

    def _set_full_name(self):
        self.full_name = self.config['full_name']

    def _set_authors(self):
        self.authors = self.config['authors']

    def _set_paper_year(self):
        self.year = self.config['year']

    def _set_description(self):
        self.descriprion = self.config['description']

    def _set_first_mention(self):
        self.first_mention = self.config['first_mention']

    def _set_which_mention(self):
        self.other_mention = self.config['other_mention']

    def _load_config_data(self):
        with open(os.path.join(os.getenv('HOME'), DATA_PATH, 'datasets', f'{self.name}', 'config.json'), 'r') as f:
            self.config = json.load(f)

    def _load_config(self):
        self._load_config_data()
        self._set_full_name()
        self._set_authors()
        self._set_description()
        self._set_first_mention()
        self._set_which_mention()

    def load(self):
        self._load_config()
        self._load_dataframe()

    
class MetaHumorDataset(AbstractHumorDataset):
    """ #TODO FIX DOCS """

    def __init__(self, name) -> None:
        super().__init__(name)
        if name != "meta_dataset":
            raise ValueError("Only meta_dataset supported by this class")
    
    def load(self, df, train_df, test_df, valid_df):
        self.df = df
        self.train_df = train_df
        self.test_df = test_df
        self.valid_df = valid_df


class MetaHumorDatasetCreator:
    """ #TODO FIX DOCS """

    def __init__(self, params, random_state=42) -> None:
        self.params = params
        self.random_state = random_state

    def create_dataset(self) -> MetaHumorDataset:
        all_df = list()
        all_train_df = list()
        all_test_df = list()
        all_valid_df = list()

        for param in self.params:
            hd = HumorDataset(name=param["name"])
            hd.load()
            hd_train = hd.get_train()
            hd_test = hd.get_test()
            hd_valid = hd.get_valid()

            if param["drop_which_dup"]:
                if "is_duplicated" in hd_train.columns.tolist():
                    hd_train = hd_train[~hd_train["is_duplicated"]]

            if param["sample_from_train"] is not None:
                hd_train = hd_train.sample(param["sample_from_train"], random_state=self.random_state)
            
            if param["sample_from_test"] is not None:
                hd_test = hd_test.sample(param["sample_from_test"], random_state=self.random_state)
            
            if param["sample_from_valid"] is not None:
                hd_valid = hd_valid.sample(param["sample_from_valid"], random_state=self.random_state)
            
            all_df.append(
                pd.concat([hd_train, hd_valid, hd_test])
            )

            all_train_df.append(
                hd_train
            )

            all_test_df.append(
                hd_test
            )

            all_valid_df.append(
                hd_valid
            )

        
        result_meta_dataset = MetaHumorDataset(name="meta_dataset")
        result_meta_dataset.load(
            df=pd.concat(all_df),
            train_df=pd.concat(all_train_df),
            test_df=pd.concat(all_test_df),
            valid_df=pd.concat(all_valid_df)
        )

        return result_meta_dataset


    

