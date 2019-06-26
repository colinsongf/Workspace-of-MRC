# coding:utf-8
# author : apollo2mars@gmail.com

import os,sys,argparse,logging
import numpy as np
import tensorflow as tf
from datasets import DuDataset, SQuADDataset
from models import dureader, squad
from metrics import bleu, rl
from utils import 1, 2, 3

logging = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        # tokenizer normal/bert/others
        tokenizer = build_tokenizer(
            fname=[opt.dataset['train'], opt.dataset['test']],
            max_seq_len=opt.max_seq_len
            dat_fname='{0}_tokenizer.dat'.format(opt.dataset_name))
        # embedding normal/bert/others

        # model init
        self.model = opt.model_class(>>>emb, self.opt)

        # dataset
        self.trainset = opt.dataset_class(opt.dataset_file['train', tokenizer])
        self.trainset = opt.dataset_class(opt.dataset_file['test', tokenizer])
        
    def _print_opts(self):
        pass
    def _reset_params(self):
        pass

    def _train(self, criterion, optimizer, train_data_loader, test_data_loader):
        
        for epoch in range(self.opt.epochs):
            logger.info('>' * 200)
            logger.info('epoch:{}'.format(epoch))
        
        iterator = train_data_loader.make_one_shot_iterator() 
        one_element = iterator.get_next()
        
        while True:
            try:
                sample_batched = self.session.run(one_element)
                inputs = sample_batched('text') 
                #inputs = sample_batched('text') 
                #inputs = sample_batched('text') 

                model = self.model
                _ = self.session.run(model.train_op, feed_dict = {})
                self.model = model

            except tf.errors.OutOfRangeError:
                break
        
        >>> [metrics] = self._eval_metrics(val_data_loader)

    def _eval_metrics(self, data_loader):
        iterator = data_loader.make_one_shot_iterator()
        one_element = iterator.get_next()

        all_outputs = []
        while True:
            try:
                sample_batched = self.session.run(one_element)
                inputs = sample_batched('text') 
                #inputs = sample_batched('text') 
                #inputs = sample_batched('text') 

                model = self.model
                outputs = self.session.run(model.output_op, feed_dict={})

                all_outputs.extend(outputs)
            except tf.errors.OutOfRangeError:
                break

        metrics_1 = 
        logger.info(metrics_1)

        return metrics_1

        
    def run(self):
        train_data_loader = tf.data.Dataset.from_tensor_slices({'text':self.trainset.text_list}).batch(self.opt.batch_size).shuffle(10000)
        test_data_loader = tf.data.Dataset.from_tensor_slices({'text':self.testset.text_list}).batch(self.opt.batch_size)
        dev_data_loader = test_data_loader 

        logger.info('>> load data done')
        logger.info('>> train data length, max length, average length')
        logger.info('>> test data length, max length, average length')

        # train and save best model
        best_model_path = self._train(None, self.opt.optimizer, train_data_loader, dev_data_loader)

        # calculate metric on test set 
        self.saver.restore(self.session, best_model_path)
        metric_1 = self._eval_metrics(test_data_loader)
        logger.info('>> metric:{.:4f},'.format(metric_1))
        

def main():
    parser = argparse.ArgumentParser()
    # dataset setting
    parser = add_argument('--dataset_name',  
    # model setting

    # optimizer setting

    # hyper parameter setting

    # 
        
    args = parser.parse_args()

    dataset_files = {
        'squad1.1':{
            'train':...,
            'test':...}
        'squad2.0':{
            'train':...,
            'test':...}
        'dureader':{
            'train':...,
            'test':...}
        'marco':{
            'train':...,
            'test':...}
        'deepmin':{
            'train':...,
            'test':...}
    }

    model_classes = {
        'bidaf': BIDAF,
        'mlstm': MLSTM,
        'bert': BERT,
        'bert_bidaf' : BERT_BIDAF
    }

    input_clos = {
        'bidaf' : ['text'],
        'mlstm' : ['']
    }



