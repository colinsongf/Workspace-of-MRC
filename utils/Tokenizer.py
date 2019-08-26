# -*- coding: utf-8 -*-
# file: data_utils.py
# author: apollo2mars <apollo2mars@gmail.com>

import os
import pickle
import numpy as np


class Tokenizer(object):
	""" Tokenizer for Machine Reading Comprehension

	1. Input : max length of context
	2. Get vocabulary dict : self.word2idx and self.idx2word
	3. Get Embedding Matrix
		if embedding matrix exits, load from exit file
		else build new embedding matrix
	"""

	def __init__(self, origin_text, max_seq_len, emb_dim, lower, emb_type, dat_fname, fname):
		self.origin_text = origin_text
		self.max_seq_len = max_seq_len
		self.emb_dim = emb_dim
		self.lower = lower
		self.emb_type = emb_type
		self.path_1 = dat_fname
		self.path_2 = fname

		self.word2idx = {}
		self.idx2word = {}
		self.vocab_embed = []

		self.embedding_info = Tokenizer.__embedding_info()
		self.__load_embedding(word2idx=self.word2idx, emb_dim=self.emb_dim, dat_fname=self.path_1, fname=self.path_2)
		self.__set_vocabulary(self.origin_text)
		self.__encode_vocab()

	@staticmethod
	def __embedding_info():
		embedding_files = {
			'Static':{
				"Word2Vec":"",
				"Glove":"",
				"Tencent":""},
			'Dynamic':{
				"BERT":"",
				"ELMo":"",
				"ERINE":"",
				"GPT-2-Chinese":"",
				"BERT-WWW":""}
		}

		return embedding_files

	def __load_embedding(self, word2idx, emb_dim, dat_fname, fname):
		if os.path.exists(dat_fname):
			embedding_matrix = pickle.load(open(dat_fname, 'rb'))
		else:
			embedding_matrix = np.zeros((len(word2idx) + 2, emb_dim))  # idx 0 and len(word2idx)+1 are all-zeros
			word_vec = Tokenizer.__get_vocabulary_embedding_vector_list(fname, word2idx=word2idx)
			for word, i in word2idx.items():
				embedding_matrix[i] = word_vec[word]
			pickle.dump(embedding_matrix, open(dat_fname, 'wb'))

		self.embedding_matrix = embedding_matrix

	def __set_vocabulary(self, input_text):
		"""
		:param text: text for generate vocabulary
		:return: null
		"""
		if self.lower:
			tmp_text = input_text.lower()

		from collections import Counter
		count = Counter(tmp_text)

		for idx, item in enumerate(count):
			self.word2idx[item] = idx + 1  # must + 1
			self.idx2word[idx + 1] = item

	def __encode_vocab(self, input_path, word2idx=None):
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
		fin = open(input_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
		word_vec = {}
		for line in fin:
			tokens = line.rstrip().split(' ')
			if word2idx is None or tokens[0] in word2idx.keys():
				word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')

		self.vocab_embed = word_vec

	@classmethod
	def __pad_and_truncate(cls, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
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
		convert text to numberical gigital features with max length, paddding
		and truncating
		"""
		if self.lower:
			text = text.lower()
		words = list(text)
		unknown_idx = len(self.word2idx)+1
		sequence = [self.word2idx[w] if w in self.word2idx else unknown_idx for w in words]
		if len(sequence) == 0:
			sequence = [0]
		if reverse:
			sequence = sequence[::-1]

		return Tokenizer.__pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


if __name__ == '__main__':
	tokenizer = Tokenizer(512, emb_type='tencent')
	text = "中文自然语言处理，Natural Language Process"
	print(tokenizer.encode(text))
