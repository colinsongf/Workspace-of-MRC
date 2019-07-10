# -*- utf-8 -*-
# author : apollo2mars@gmail.com
"""
do data preprocess for train_keras.py and test_keras.py
save the result
"""

from utils.squad_preprocess import context_question_text_preprocess
import numpy as np


class DataHelper():
    def __init__(self, args):
        self.len_cnt_padding = args.context_padding_length
        self.len_qn_padding = args.query_padding_length

        self.embedding_matrix = None
        self.vocab_size = None
        self.pad_txt_train_cnt = None
        self.pad_txt_train_qry = None
        self.pad_txt_dev_cnt = None
        self.pad_txt_dev_qry = None
        self.idx_train_beg = None
        self.idx_train_end = None
        self.idx_dev_beg = None
        self.idx_dev_end = None

    def run(self):
        embedding_matrix, vocab_size, pad_txt_train_cnt, pad_txt_train_qry, pad_txt_dev_cnt, pad_txt_dev_qry, \
            idx_train_beg, idx_train_end, idx_dev_beg, idx_dev_end = context_question_text_preprocess(self.len_cnt_padding, self.len_qn_padding)

        """
        transform data to np format
        """
        idx_train_beg = np.asarray(idx_train_beg)
        idx_dev_beg = np.asarray(idx_dev_beg)
        idx_train_end = np.asarray(idx_train_end)
        idx_dev_end = np.asarray(idx_dev_end)

        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.pad_txt_train_cnt = pad_txt_train_cnt
        self.pad_txt_train_qry = pad_txt_train_qry
        self.pad_txt_dev_cnt = pad_txt_dev_cnt
        self.pad_txt_dev_qry = pad_txt_dev_qry
        self.idx_train_beg = idx_train_beg
        self.idx_train_end = idx_train_end
        self.idx_dev_beg = idx_dev_beg
        self.idx_dev_end = idx_dev_end
