import os
import tempfile
from zipfile import ZipFile
import copy

import gdown
import numpy as np
import requests

class MidDataset:

    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.non_unique_words = 1


SUPPORTED_DATASETS = ['pun_of_the_day', 'one_liners', 'reddit_jokes_last_laught',
                      'short_jokes', 'funlines_v1', 'human_microedit_v1',
                      'funlines_v2', 'human_microedit_v2', 'unfun_me',
                      'semeval_2021_task_7', 'semeval_2017_task_7', 'the_naughtyformer',
                      'funlines_and_human_microedit_paper_2023', 'onion_or_not', 'meta_dataset', 'comb'
                      ]

SUPPORTED_CONVERSATIONAL_DATASETS = ['alice', 'curiousity', 'fig_qa_end', 'fig_qa_start', 'friends', 'irony', 'three_men', 'walking_dead']

not_used_in_paper_2023 = ['funlines_v1', 'funlines_v2', 'human_microedit_v1',
                          'human_microedit_v2', 'meta_dataset'
                         ]

HRI_PAPER_2023_DATASETS = list(
    set(SUPPORTED_DATASETS).difference(set(not_used_in_paper_2023))
)

SUPPORTED_DATASETS = sorted(SUPPORTED_DATASETS)
HRI_PAPER_2023_DATASETS = sorted(HRI_PAPER_2023_DATASETS)

# Simple logic of moving to the end of the largest sets
SUPPORTED_DATASETS.append(SUPPORTED_DATASETS.pop(SUPPORTED_DATASETS.index('short_jokes')))
SUPPORTED_DATASETS.append(SUPPORTED_DATASETS.pop(SUPPORTED_DATASETS.index('the_naughtyformer')))
HRI_PAPER_2023_DATASETS.append(HRI_PAPER_2023_DATASETS.pop(HRI_PAPER_2023_DATASETS.index('short_jokes')))
HRI_PAPER_2023_DATASETS.append(HRI_PAPER_2023_DATASETS.pop(HRI_PAPER_2023_DATASETS.index('the_naughtyformer')))

USER_HOME = os.getenv('HOME')
DATA_PATH = os.path.join(USER_HOME, 'hri_tools_data/')

BAD_WORDS_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"


def download():
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        url = os.getenv('HRI_URL')
        password = os.getenv('HRI_PASSWORD')
        password = bytes(password, 'utf-8')
        
        output = os.path.join(tmpdirname, 'dataset.zip')
        gdown.download(url, output, quiet=False, fuzzy=True)

        with ZipFile(output) as zf:
            zf.extractall(
                path=DATA_PATH,
                pwd=password
            )
        
        print('DONE!')


def calc_divergence(first_dataset, second_dataset) -> float:
    '''
    Function for calculate different divergence of two datasets
    '''

    def _non_symmetric_kl_divergence(first_dataset, second_dataset):
        
        elements_to_add_for_second = list(
            set(first_dataset.vocab.keys()) - set(second_dataset.vocab.keys())
        )

        edited_second_vocab = copy.deepcopy(second_dataset.vocab)
        non_unique_words_second = second_dataset.non_unique_words
        
        for k in elements_to_add_for_second:
            edited_second_vocab[k] = 0
        
        for k in edited_second_vocab.keys():
            edited_second_vocab[k] = (edited_second_vocab[k] + 1/3) / (non_unique_words_second + 1/3 * len(edited_second_vocab))

        first_vocab = copy.deepcopy(first_dataset.vocab)
        non_unique_words_first = first_dataset.non_unique_words

        for k in first_vocab.keys():
            first_vocab[k] = (first_vocab[k] + 1/3) / (non_unique_words_first + 1/3 * len(first_vocab))
        
        result = 0
        for k in first_vocab.keys():
            result += first_vocab[k] * np.log2(first_vocab[k] / edited_second_vocab[k])

        return result
    
    def _build_middle_vocab(first_dataset, second_dataset):
        elements_to_add_for_second = list(
            set(first_dataset.vocab.keys()) - set(second_dataset.vocab.keys())
        )

        elements_to_add_for_first = list(
            set(second_dataset.vocab.keys()) - set(first_dataset.vocab.keys())
        )

        edited_second_vocab = copy.deepcopy(second_dataset.vocab)
        non_unique_words_second = second_dataset.non_unique_words
        
        for k in elements_to_add_for_second:
            edited_second_vocab[k] = 0
        
        for k in edited_second_vocab.keys():
            edited_second_vocab[k] = (edited_second_vocab[k] + 1/3) / (non_unique_words_second + 1/3 * len(edited_second_vocab))

        edited_first_vocab = copy.deepcopy(first_dataset.vocab)
        non_unique_words_first = first_dataset.non_unique_words

        for k in elements_to_add_for_first:
            edited_first_vocab[k] = 0
        
        for k in edited_first_vocab.keys():
            edited_first_vocab[k] = (edited_first_vocab[k] + 1/3) / (non_unique_words_first + 1/3 * len(edited_first_vocab))
        
        result = dict()
        for k in edited_first_vocab.keys():
            result[k] = (edited_first_vocab[k] + edited_second_vocab[k]) * 0.5

        return result

    if not(first_dataset.is_vocab_built() and second_dataset.is_vocab_built()):
        raise ValueError('You shoud build vocab!!!')

    if not(first_dataset.is_preprocessed() and second_dataset.is_preprocessed()):
        raise ValueError('You shoud preprocesse!!!')

    symmetric_kl = _non_symmetric_kl_divergence(first_dataset, second_dataset) + _non_symmetric_kl_divergence(second_dataset, first_dataset)
    js_divergence = None
    
    # m = _build_middle_vocab(first_dataset, second_dataset)
    # middle_dataset = MidDataset(m)
    # js_divergence = 0.5 * _non_symmetric_kl_divergence(first_dataset, middle_dataset) + 0.5 * _non_symmetric_kl_divergence(second_dataset, middle_dataset)

    return {
        'Symmetrised KL divergence': symmetric_kl,
        'Jensen–Shannon divergence': js_divergence
    }





def calc_divergence_between_target(first_dataset) -> float:
    '''
    Function for calculate different divergence true and false set
    '''

    second_dataset = copy.deepcopy(first_dataset)

    first_dataset.df = first_dataset.df[first_dataset.df['label']==1]

    second_dataset.df = second_dataset.df[second_dataset.df['label']==0]

    first_dataset.run_preprocessing()
    second_dataset.run_preprocessing()

    first_dataset.build_vocab()
    second_dataset.build_vocab()


    def _non_symmetric_kl_divergence(first_dataset, second_dataset):
        
        elements_to_add_for_second = list(
            set(first_dataset.vocab.keys()) - set(second_dataset.vocab.keys())
        )

        edited_second_vocab = copy.deepcopy(second_dataset.vocab)
        non_unique_words_second = second_dataset.non_unique_words
        
        for k in elements_to_add_for_second:
            edited_second_vocab[k] = 0
        
        for k in edited_second_vocab.keys():
            edited_second_vocab[k] = (edited_second_vocab[k] + 1/3) / (non_unique_words_second + 1/3 * len(edited_second_vocab))

        first_vocab = copy.deepcopy(first_dataset.vocab)
        non_unique_words_first = first_dataset.non_unique_words

        for k in first_vocab.keys():
            first_vocab[k] = (first_vocab[k] + 1/3) / (non_unique_words_first + 1/3 * len(first_vocab))
        
        result = 0
        for k in first_vocab.keys():
            result += first_vocab[k] * np.log2(first_vocab[k] / edited_second_vocab[k])

        return result

    symmetric_kl = _non_symmetric_kl_divergence(first_dataset, second_dataset) + _non_symmetric_kl_divergence(second_dataset, first_dataset)
    
    return symmetric_kl



def calc_vocab_for_labels(first_dataset):

    second_dataset = copy.deepcopy(first_dataset)

    first_dataset.df = first_dataset.df[first_dataset.df['label']==1]

    second_dataset.df = second_dataset.df[second_dataset.df['label']==0]

    first_dataset.run_preprocessing()
    second_dataset.run_preprocessing()

    first_dataset.build_vocab()
    second_dataset.build_vocab()

    print(first_dataset.name)
    print('positive target')
    print("vocab_size", first_dataset.vocab_size, "non_unique_words", first_dataset.non_unique_words, "token ratio", first_dataset.vocab_size/first_dataset.non_unique_words)
    print('negative target')
    print("vocab_size", second_dataset.vocab_size, "non_unique_words", second_dataset.non_unique_words, "token ratio", second_dataset.vocab_size/second_dataset.non_unique_words)
    return ''


def get_bad_words():
    resp = requests.get(BAD_WORDS_URL)
    return set(resp.content.decode().split("\n")[:-1])