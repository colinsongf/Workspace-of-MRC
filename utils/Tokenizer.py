# -*- coding: utf-8 -*-
# file: data_utils.py
# author: apollo2mars <apollo2mars@gmail.com>

import os
import pickle
import numpy as np

embedding_files = {
	'Static':{
		"Word2Vec":"",
		"Glove":"",
		"Tencent":""},
	'Dynamic':{
		"BERT":"",
		"ELMo":""}
}


class Tokenizer(object):
	""" Tokenizer for Machine Reading Comprehension

	1. Input : max length of context
	2. Get vocabulary dict : self.word2idx and self.idx2word
	3. Get Embedding Matrix
		if embedding matrix exits, load from exit file
		else build new embedding matrix
	"""

	def __init__(self, max_seq_len, lower=True, emb_type="tencent", ):
		self.lower = lower
		self.max_seq_len = max_seq_len
		self.emb_type = emb_type
		self.word2idx = {}
		self.idx2word = {}

		self.get_vocabulary()

	@staticmethod
	def _pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
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

	@staticmethod
	def _load_word_vec(input_path, word2idx=None):
		"""
		Read

		:param input_path:
		:param word2idx:
		:return:
		"""
		fin = open(input_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
		word_vec = {}
		for line in fin:
			tokens = line.rstrip().split(' ')
			if word2idx is None or tokens[0] in word2idx.keys():
				word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')

			return word_vec

	def fit_on_teixt(self, text):
		if self.lower:
			text = text.lower()

		from collections import Counter
		count = Counter(text)

		for idx, item in enumerate(count):
			self.word2idx[item] = idx + 1 # must + 1
			# self.idx2word[idx + 1] = item

	def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
		if self.lower:
			text = text.lower()
		words = list(text)
		unknownidx = len(self.word2idx)+1
		sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]

		if len(sequence) == 0:
			sequence = [0]
		if reverse:
			sequence = sequence[::-1]

		return _pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

	def build_static_embedding_matrix(word2idx, embed_dim, dat_fname, fname):
		if os.path.exists(dat_fname):
			embedding_matrix = pickle.load(open(dat_fname, 'rb'))
		else:
			embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
			word_vec = _load_word_vec(fname, word2idx=word2idx)
			for word, i in word2idx.items():
				embedding_matrix[i] = word_vec[word]
			pickle.dump(embedding_matrix, open(dat_fname, 'wb'))

		return embedding_matrix

	def build_dynamic_embedding():
		pass






