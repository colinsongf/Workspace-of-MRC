# coding:utf-8
# author:Apollo2Mars@gmail.com
# function:dataset loader for squad2


class SQuAD2Dataset():
	def __init__(self, opt):
		self.opt = opt
		self.max_query_len = opt.max_query_len
		self.max_passage_len = opt.max_passage_len
		self.max_passage_num = opt.max_passage_num

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
