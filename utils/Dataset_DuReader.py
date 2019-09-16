"""
This module implements data process strategies.
"""

import json
import logging
import numpy as np
from collections import Counter


class Dataset_DuReader(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len, train_files=[], dev_files=[], test_files=[]):
        # 共6个参数
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num   # BRCDataset init
        self.max_p_len = max_p_len   # BRCDataset init
        self.max_q_len = max_q_len   # BRCDataset init

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            data_set = []
            for lidx, line in enumerate(fin):  # 读一个文件
                sample = json.loads(line.strip())  # 读单条数据
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
                        para_infos.sort(key=lambda x: (-x[1], x[2]))  ## 按照 recall_wrt_question 排序， 相同的recall_wrt_question 按照 para_tokens 排序

                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:  # 遍历每个段落最正确的答案
                            fake_passage_tokens += para_info[0]

                        sample['passages'].append({'passage_tokens': fake_passage_tokens})

                # sample['answer_spans']
                # sample['answer_passages']
                # sample['question_tokens']
                # sample['passage'] [passage_tokens train : segmented_paragraphs(right answer),
                #                                   other : fake_answer_tokens]
                data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        # 描述批次数据的 数据结构，当前这块可能有问题
        batch_data = {'raw_data': [data[i] for i in indices],  # select batch data
                      # 'question_token_ids': [],
                      'question_length': [],
                      # 'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': [],
                      'q_input_ids': [],
                      'q_input_mask': [],
                      'q_segment_ids': [],
                      'p_input_ids': [],
                      'p_input_mask': [],
                      'p_segment_ids': []
                      }
        # 最大段落数
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        # 遍历batch
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):

                    q_input_ids = sample['question_token_ids']
                    if len(q_input_ids) > self.max_q_len:
                        q_input_ids = q_input_ids[0:self.max_q_len]
                    q_input_mask = [1] * len(q_input_ids)
                    q_segment_ids = [0] * len(q_input_ids)

                    p_input_ids = sample['passages'][pidx]['passage_token_ids']
                    if len(p_input_ids) > self.max_p_len:
                        p_input_ids = p_input_ids[0:self.max_p_len]
                    p_input_mask = [1] * len(p_input_ids)
                    p_segment_ids = [0] * len(p_input_ids)

                    batch_data['question_length'].append(min(len(q_input_ids), self.max_q_len))
                    batch_data['passage_length'].append(min(len(p_input_ids), self.max_p_len))

                    # padding
                    while len(q_input_ids) < self.max_q_len:
                        q_input_ids.append(0)
                        q_input_mask.append(0)
                        q_segment_ids.append(0)

                    while len(p_input_ids) < self.max_p_len:
                        p_input_ids.append(0)
                        p_input_mask.append(0)
                        p_segment_ids.append(0)

                    assert len(p_input_ids) == self.max_p_len
                    assert len(p_input_mask) == self.max_p_len
                    assert len(p_segment_ids) == self.max_p_len

                    assert len(q_input_ids) == self.max_q_len
                    assert len(q_input_mask) == self.max_q_len
                    assert len(q_segment_ids) == self.max_q_len

                    batch_data['q_input_ids'].append(q_input_ids)
                    batch_data['q_input_mask'].append(q_input_mask)
                    batch_data['q_segment_ids'].append(q_segment_ids)

                    batch_data['p_input_ids'].append(p_input_ids)
                    batch_data['p_input_mask'].append(p_input_mask)
                    batch_data['p_segment_ids'].append(p_segment_ids)

                else:
                    batch_data['question_length'].append(0)
                    batch_data['passage_length'].append(0)

                    batch_data['q_input_ids'].append([0] * self.max_q_len)
                    batch_data['q_input_mask'].append([0] * self.max_q_len)
                    batch_data['q_segment_ids'].append([0] * self.max_q_len)

                    batch_data['p_input_ids'].append([0] * self.max_p_len)
                    batch_data['p_input_mask'].append([0] * self.max_p_len)
                    batch_data['p_segment_ids'].append([0] * self.max_p_len)

        # padding data for batch_data question_token_ids, passage_token_ids
        # batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        # start_id
        # end_id
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = self.max_p_len * sample['answer_passages'][0]
                # gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])

            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)

        with open('batch_data.txt', encoding='utf-8', mode='w') as f:
            f.write('\n\n batch_data \n\n')
            f.write(str(batch_data))
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        # 长度是变化的，暂时不使用
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)


def build_dataset():


if __name__ == '__main__':
    prefix_path = 'corpus/dureader/preprocessed/preprocessed/'
    train_files = [prefix_path + 'trainset/search.train.json']
    dev_files = [prefix_path + 'devset/search.train.json']
    test_files = [prefix_path + 'testset/search.train.json']

    dataset = DuReaderDataset(max_p_num=5, max_p_len=5, max_q_len=5, train_files=train_files, dev_files=dev_files, test_files=test_files)
