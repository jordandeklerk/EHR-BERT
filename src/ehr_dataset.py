from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import random
import copy
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def random_word(tokens, vocab):
    for i, _ in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"
            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(vocab.word2idx.items()))[0]
            else:
                pass
        else:
            pass

    return tokens


class EHRTokenizer(object):
    """Runs end-to-end tokenization"""

    def __init__(self, data_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.proc_voc = self.add_vocab(os.path.join('./vocab', 'proc-vocab.txt'))
        self.dx_voc = self.add_vocab(os.path.join('./vocab', 'dx-vocab.txt'))

    def add_vocab(self, vocab_file):
        voc = self.vocab
        specific_voc = Voc()
        with open(vocab_file, 'r') as fin:
            for code in fin:
                voc.add_sentence([code.rstrip('\n')])
                specific_voc.add_sentence([code.rstrip('\n')])
        return specific_voc

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens


class EHRDataset(Dataset):
    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len

        self.sample_counter = 0

        def transform_data(data):
            """
            :param data: raw data form
            :return: {subject_id, [adm, 2, codes]},
            """
            admissions = []
            for _, row in data.iterrows():
                admission = [list(row['ICD9_CODE']), list(row['ICD9_PROC_CODES'])]
                admissions.append(admission)
            return admissions

        self.admissions = transform_data(data_pd)

    def __len__(self):
        return len(self.admissions)

    def __getitem__(self, item):
        cur_id = item
        adm = copy.deepcopy(self.admissions[item])

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        y_dx = np.zeros(len(self.tokenizer.dx_voc.word2idx))
        y_proc = np.zeros(len(self.tokenizer.proc_voc.word2idx))

        # Handle diagnosis codes
        for item in adm[0]:
            if item in self.tokenizer.dx_voc.word2idx:
                y_dx[self.tokenizer.dx_voc.word2idx[item]] = 1
            else:
                print(f"Warning: Diagnosis token {item} not found in dx_voc.")
        for item in adm[1]:
            if item in self.tokenizer.proc_voc.word2idx:
                y_proc[self.tokenizer.proc_voc.word2idx[item]] = 1
            else:
                print(f"Warning: Token {item} not found in proc_voc.")

        # replace tokens with [MASK]
        adm[0] = random_word(adm[0], self.tokenizer.dx_voc)
        adm[1] = random_word(adm[1], self.tokenizer.proc_voc)

        # extract input and output tokens
        input_tokens = []  # (2*max_len)
        input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))

        # convert tokens to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        if cur_id < 5:
            logger.info("*** Example ***")
            logger.info("input tokens: %s" % " ".join(
                [str(x) for x in input_tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))

        cur_tensors = (torch.tensor(input_ids, dtype=torch.long).view(-1, self.seq_len),
                       torch.tensor(y_dx, dtype=torch.float),
                       torch.tensor(y_proc, dtype=torch.float))

        return cur_tensors



def load_dataset(data):
    data_dir = './data'
    max_seq_len = 55

    # load tokenizer
    tokenizer = EHRTokenizer(data_dir)

    # load data
    data = pd.read_pickle(os.path.join(data_dir, 'data-comb-visit.pkl'))

    # load trian, eval, test data
    ids_file = [os.path.join(data_dir, 'train-id.txt'),
                os.path.join(data_dir, 'eval-id.txt'),
                os.path.join(data_dir, 'test-id.txt')]

    def load_ids(data, file_name):
        ids = []
        with open(file_name, 'r') as f:
            for line in f:
                ids.append(line.rstrip('\n'))
        # print("Loaded IDs from", file_name, ":", ids)
        filtered_data = data[data['SUBJECT_ID'].isin(ids)].reset_index(drop=True)
        print("Filtered data shape:", filtered_data.shape)
        return filtered_data

    train_dataset, eval_dataset, test_dataset = tuple(map(lambda x: EHRDataset(load_ids(data, x), tokenizer, max_seq_len), ids_file))

    return tokenizer, train_dataset, eval_dataset, test_dataset



def split_dataset(data_path):
    """
    Splits the dataset into training, evaluation, and testing datasets using train_test_split.
    Writes the IDs for each split into separate files.

    Parameters:
    - data_path (str): The path to the pickle file containing the DataFrame.

    Outputs:
    - Files containing the IDs for the train, eval, and test datasets.
    """
    np.random.seed(315)  # Setting seed for reproducibility

    data = pd.read_pickle(data_path)
    sample_id = data['SUBJECT_ID'].unique()

    # Splitting the data into train (67%) and temp (33%)
    train_id, temp_id = train_test_split(sample_id, test_size=1/3, random_state=315)

    # Splitting the temp into eval (50% of temp, 16.5% of total) and test (50% of temp, 16.5% of total)
    eval_id, test_id = train_test_split(temp_id, test_size=0.5, random_state=315)

    def ids_to_file(ids, file_name):
        with open(file_name, 'w') as fout:
            for item in ids:
                fout.write(str(item) + '\n')

    ids_to_file(train_id, './data/train-id.txt')
    ids_to_file(eval_id, './data/eval-id.txt')
    ids_to_file(test_id, './data/test-id.txt')

    print(f'train size: {len(train_id)}, eval size: {len(eval_id)}, test size: {len(test_id)}')


path = './data/data-comb-visit.pkl'
split_dataset(path)


print("Loading Dataset...")
tokenizer, train_dataset, eval_dataset, test_dataset = load_dataset(path)
train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=64)
eval_dataloader = DataLoader(eval_dataset,
                                 sampler=SequentialSampler(eval_dataset),
                                 batch_size=64)
test_dataloader = DataLoader(test_dataset,
                                 sampler=SequentialSampler(test_dataset),
                                 batch_size=64)