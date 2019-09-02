# -*- coding: utf-8 -*-
# file: data_utils.py
# author: apollo2mars <apollo2mars@gmail.com>

import pickle
import numpy as np

import os,sys
from os import path
sys.path.append(path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Tokenizer(object):
    """ Tokenizer for Machine Reading Comprehension

    1. Input : max length of context
    2. Get vocabulary dict : self.word2idx and self.idx2word
    3. Get Embedding Matrix
        if embedding matrix exits, load from exit file
        else build new embedding matrix
    """
    def __init__(self, origin_file, max_seq_len, emb_type, dat_fname):

        self.max_seq_len = max_seq_len
        self.emb_type = emb_type.lower()
        self.dat_path = dat_fname

        self.lower = True

        lines = open(origin_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        self.fit_text = ' '.join(lines)

        self.embedding_info = {}
        self.word2idx = {}
        self.idx2word = {}
        self.word2vec = {}  # {"Apple":[1,2,2,3,5], "Book":"1,2,1,1,1"}
        self.embedding_matrix = []

        self.__set_embedding_info()
        self.__set_vocabulary(input_text=self.fit_text)
        self.__set_word2vec(embedding_path=self.embedding_files['Static'][self.emb_type],
                            word2idx=self.word2idx)
        self.__set_embedding_matrix(word2idx=self.word2idx, dat_fname=self.dat_path)
        
    def __set_embedding_info(self):
        embedding_files = {
            'Static':{
                "Word2Vec":"",
                "Glove":"",
                "tencent":"../resources/Tencent_AILab_ChineseEmbedding.txt"
            },
            'Dynamic':{
                "BERT":"",
                "ELMo":"",
                "ERINE":"",
                "GPT-2-Chinese":"",
                "BERT-WWW":""
            }
        }
        
        self.embedding_files = embedding_files

    def __set_vocabulary(self, input_text):
        """
        :param input_text: text for generate vocabulary
        :return: null
        """
        if self.lower:
            tmp_text = input_text.lower()

        from collections import Counter
        count = Counter(tmp_text)

        for idx, item in enumerate(count):
            self.word2idx[item] = idx + 1  # must + 1
            self.idx2word[idx + 1] = item

    def __set_word2vec(self, embedding_path, word2idx=None):
        """
        :param input_path: staic embedding file, for example("Glove.5b.300d")
                , [0.2,0.6,..,0.2]
                Apple [0.3,0.3,..,0.7]
                Bob [0.3,0.4,..,0.7]
                Car [0.5,0.4,..,0.7]
                Do [0.8,0.4,..,0.7]
                Eat [0.9,0.4,..,0.7]
                ...
                Zip [0.3,0.6,..,0.7]
        :param word2idx: vocabulary for current task  [list]
                input file : Bob Eat Apple
                [Apple, Eat, Apple]
        :return: embedding vector list for vocabury
                [[0.3,0.4,..,0.7]
                [0.9,0.4,..,0.7]
                [0.3,0.3,..,0.7]]

        get embeddding vector list from embedding matrix by vovabulary

        """
        fin = open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        word2vec = {}
        for line in fin:
            tokens = line.rstrip().split(' ') 
            if tokens[0] in word2idx.keys():
                print("="*20, "go in")
                word2vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        
        self.word2vec = word2vec

    def __set_embedding_matrix(self, word2idx, dat_fname):
        if os.path.exists(dat_fname):
            print("use exist embedding file")
            embedding_matrix = pickle.load(open(dat_fname, 'rb'))
            print(embedding_matrix)
        elif self.emb_type == 'random':
            embedding_matrix = np.zeros((len(word2idx) + 2, 300))  # idx 0 and len(word2idx)+1 are all-zeros
            pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
        elif self.emb_type == 'tencent':
            print("************, do tencent embedding")
            embedding_matrix = np.zeros((len(word2idx) + 2, 200))  # idx 0 and len(word2idx)+1 are all-zeros

            for word, i in word2idx.items():
                embedding_matrix[i] = self.word2vec[word]
            pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
        elif self.emb_type == 'bert':
            pass
       
        self.embedding_matrix = embedding_matrix
        

    def __pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        x = (np.ones(maxlen) * value).astype(dtype)

        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
            trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x


    def encode(self, text, reverse=False, padding='post', truncating='post'):
        """
        :param text:
        :param reverse:
        :param padding:
        :param truncating:
        :return: convert text to numberical digital features with max length, paddding
        and truncating
        """
        if self.lower:
            text_lower = text.lower()
        words = list(text_lower)
        unknown_idx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknown_idx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]

        tmp_list = self.__pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        return [ self.embedding_matrix[item] for item in tmp_list]


if __name__ == '__main__':
    emb_type = 'tencent'
    tokenizer = Tokenizer(origin_file = "a.txt", max_seq_len=512, emb_type=emb_type, dat_fname=emb_type+"_tokenizer.dat")
    text = "中文自然语言处理，Natural Language Process"
    print(tokenizer.word2idx)
    print(tokenizer.idx2word)
    print(tokenizer.encode("中文"))
