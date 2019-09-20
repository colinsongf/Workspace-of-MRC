"""
This module implements data process strategies.
"""

import json
import os
import logging
import numpy as np
from collections import Counter
from  utils.Tokenizer import build_tokenizer


class Dataset_DuReader(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, tokenizer,  max_p_num, max_p_len, max_q_len, train_files=[], dev_files=[], test_files=[]):
        # 共6个参数
        self.logger = logging.getLogger("brc")
        self.tokenizer = tokenizer
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset_raw(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset_raw(dev_file)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset_raw(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset_raw(self, data_path, train=False):
        """
        input data structure : [item1, item2,... , ]
        "documents" : [{"is_selected": True, "title":"...", "paragraphs":[]},
                      {"is_selected": False, "title":"...", "paragraphs":[]},
                      ...
                      {"is_selected": False, "title":"...", "paragraphs":[]}]
        "question" : "..."
        "answers" : ["...", "...", ..., "..."]
        "question_type" : "DESCRIPTION" or "Entity" or "Yes_No"
        "question_id" : 11111,
        "fact_or_opinion" : "FACT"

        :param data_path:
        :param train:
        :return: document, quesiton, answer, question_type, fact_or_opinion
        """

        with open(data_path) as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = {}
                item = json.loads(line.strip())

                if item['question_type'] != "DESCRIPTION":
                    continue

                # document
                if train:
                    sample['document'] = item['document']['most_related_para']
                else:
                    sample['document'] = item['document']

                # question
                sample['question'] = item['question']

                # answer
                sample['answer'] = item['answers'][0]

                data_set.append(sample)

        """
        output data structure
        [sample1, sample2, ..., ]
        """

    def _load_dataset_preprocess(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            """
            input structure
            answer_spans
            answer_passages
            question_tokens
            passages [ passages : ...
                       is_selected : ...]
            """
            data_set = []
            for lidx, line in enumerate(fin):  # 读一个文件
                sample = json.loads(line.strip())  # 读单条数据

                # 不满足条件的数据清理掉

                if train:
                    if len(sample['answer_spans']) == 0:  # 没有答案这个字段
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:  # answer 太长，跳过
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']   #答案的正文 统一到answer_passages 这个字段

                sample['question_tokens'] = sample['segmented_question']  #问题的正文统一到 question_tokens 的字段

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):  # 遍历正文的所有自然段
                    if train:
                        most_related_para = doc['most_related_para']  # 训练的时候 只使用了 最相关的自然段 分词之后的结果
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected']}
                        )
                    else:  # 结果计算
                        para_infos = []  # 记录各个para 的结果
                        for para_tokens in doc['segmented_paragraphs']:   # 所有的paragraphs 的 segmented 结果 [[token list of para1], [token list of para2], [token list of para3]]
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())  # 统计相同的词
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)

                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))  # 按照 recall_wrt_question 排序， 相同的recall_wrt_question 按照 para_tokens 排序

                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:  # 遍历每个段落最正确的答案
                            fake_passage_tokens += para_info[0]

                        sample['passages'].append({'passage_tokens': fake_passage_tokens})

                data_set.append(sample)
        return data_set

    def __pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        """
        :param sequence:
        :param maxlen:
        :param dtype:
        :param padding:
        :param truncating:
        :param value:
        :return: sequence after padding and truncate
        """
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

    def _encode_text_sequence(self, text, max_seq_len, do_padding, do_reverse):
        """
        :param text:
        :return: convert text to numberical digital features with max length, paddding
        and truncating
        """
        words = list(text)

        sequence = [self.word2idx[w] if w in self.word2idx else
                    self.word2idx['<UNK>'] for w in words]

        if len(sequence) == 0:
            sequence = [0]
        if do_reverse:
            sequence = sequence[::-1]

        if do_padding:
            sequence = self.__pad_and_truncate(sequence, max_seq_len, value=0)

        return sequence

    def convert_text_to_index(self):
        for sample in self.train_set:
            # question encode
            sample['question'] = self._encode_text_sequence(sample['question'], self.max_q_len, True, False)

            # answer encode
            sample['answer'] = [self._encode_text_sequence(item) for item in sample['answers']]

            # documents encode
            for document in sample['documents']:
                document['title'] = self._encode_text_sequence(document['title']);
                document['paragraphs'] = [self._encode_text_sequence(item) for item in document['paragraphs']]


if __name__ == '__main__':

    prefix_path = 'corpus/dureader'
    dataset_type = 'raw'
    train_files = [os.path.join(prefix_path, dataset_type, 'trainset/search.train.json')]
    dev_files = [os.path.join(prefix_path, dataset_type, 'devset/search.train.json')]
    test_files = [os.path.join(prefix_path, dataset_type, 'testset/search.train.json')]

    tokenizer = build_tokenizer(corpus_files=[train_files, test_files, dev_files], corpus_type='MRC', embedding_type='tencent')

    dataset = Dataset_DuReader(tokenizer, max_p_num=5, max_p_len=5, max_q_len=5, train_files=train_files, dev_files=dev_files, test_files=test_files)

