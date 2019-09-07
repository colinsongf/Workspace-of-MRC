# coding:utf-8
# author:Apollo2Mars@gmail.com
# function:dataset loader for squad2

import tensorflow as tf

class DatasetSQuAD2():
	def __init__(self, opt, tokenizer):
		self.opt = opt

		self.tokenizer = tokenizer

		self.max_query_len = opt.max_query_len
		self.max_passage_len = opt.max_passage_len
		self.max_passage_num = opt.max_passage_num

		train_data_loader = tf.data.Dataset.from_tensor_slices({'text': self.trainset.text_list}).batch(
			self.opt.batch_size).shuffle(10000)
		test_data_loader = tf.data.Dataset.from_tensor_slices({'text': self.testset.text_list}).batch(
			self.opt.batch_size)
		dev_data_loader = test_data_loader


	def _load_dataset(self):
		pass

	def _load_tokenizer(self):
		pass

	def _load_embedder(self):
		pass

	def _one_mini_batch(self):
		pass

	def gen_one_mini_batch(self):
		pass


