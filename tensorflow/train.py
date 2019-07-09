# coding:utf-8
# author : apollo2mars@gmail.com

import os,sys,argparse,logging
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import numpy as np
import tensorflow as tf
from utils.DatasetDuReader import DuReaderDataset
from utils.DatasetSQuAD2 import
from models import dureader, squad2, squad
from metrics import bleu, rl
from utils.data_utils import *

from time import strftime, localtime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))



class Instructor:
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model:
            tokenizer = build_tokenizer_bert(fname=[opt.dataset['train'], opt.dataset['test']], max_seq_len=opt.max_seq_len, dat_fname='{0}_tokenizer.dat'.format(opt.dataset_name))
            bert =
        else :
            tokenizer = build_tokenizer(fname=[opt.dataset['train'], opt.dataset['test']], max_seq_len=opt.max_seq_len, dat_fname='{0}_tokenizer.dat'.format(opt.dataset_name))
            embedding = build_embedding_matrix(word2idx=tokenizer.word2idx, embed_dim=opt.emb_dim, dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.emb_dim), opt.dataset))
            self.model = opt.model_class(embedding, self.opt)

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
    parser.add_argument('--model', type=str, default = 'bidaf')
    parser.add_argument('--dataset', type=str, default = 'dureader')
    parser.add_argument('--optimizer', type=str, default = 'dureader')
    parser.add_argument('--initializer', type=str, default = 'dureader')
    parser.add_argument('--learning_rate', type=float, default = 5e-5)
    parser.add_argument('--epoch', type=int, default = 10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--multi_gpu', type=str, default='0,1,2,3')

    parser.add_argument('--pretrain_bert_name', type=str, default='chinese')
    parser.add_argument('--emb_dim', type=int, default=300)

    args = parser.parse_args()

    dataset_files = {
        'squad1.1':{
            'train':'',
            'test':''},
        'squad2.0':{
            'train':'corpus/squad2/train-v2.0.json',
            'test':'corpus/squad2/dev-v2.0.json'},
        'dureader':{
            'train':'',
            'test':''},
        'marco':{
            'train':'',
            'test':''},
        'deepmin':{
            'train':'',
            'test':}
    }

    model_classes = {
        'bidaf': BIDAF,
        'mlstm': MLSTM,
        'bert': BERT,
        'bert_bidaf' : BERT_BIDAF
    }

    input_cols = {
        'squad2.0' : ['text']
    }

    optimizers = {
        'adam':tf.train.AdamOptimizer
    }

    args.dataset = dataset_files[args.dataset]
    args.model = model_classes[args.model]
    args.input_cols = input_cols[args.input_cols]
    args.optimizer = optimizers[args.optimizer]

    log_file = './outputs/logs/{}-{}-{}.log'.format(args.dataset_file, args.model_class, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()

if __name__ == '__main__':
    main()



