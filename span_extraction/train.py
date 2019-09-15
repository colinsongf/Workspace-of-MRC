# coding:utf-8
# author : apollo2mars@gmail.com

import sys,argparse,logging
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import tensorflow as tf
<<<<<<< HEAD
import  numpy as np

=======
>>>>>>> 6b945f153e4c06b55dc990fab32bcf6ef7a49337
from time import strftime, localtime

from utils.Tokenizer import build_tokenizer
from span_extraction.models import BIDAF
<<<<<<< HEAD
from utils.Tokenizer import build_tokenizer
=======
from utils.DatasetSQuAD2 import DatasetSQuAD2
>>>>>>> 6b945f153e4c06b55dc990fab32bcf6ef7a49337

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

<<<<<<< HEAD
        # pre-processing : tokenizer
        tokenizer = build_tokenizer(fname=[opt.dataset['train'], opt.dataset['test']], max_seq_len=opt.max_seq_len, dat_fname='{0}_tokenizer.dat'.format(opt.dataset_name))

        # build model
        self.model = opt.model_class(self.opt, tokenizer)

        # build dataset
        self.train_set, self.dev_set, self.test_set, self.predict_set = opt.dataset()
=======
        # build tokenizer
        tokenizer = build_tokenizer(corpus_files=[opt.dataset['train'], opt.dataset['test']], max_seq_len=512, corpus_type='qa', embedding_type='tencent')

        # build model and session
        self.model = BIDAF(self.opt, tokenizer)
        self.session = self.model.session

        # dataset
        self.trainset = opt.dataset_class(opt.dataset['train'], tokenizer)
        self.testset = opt.dataset_class(opt.dataset['test'], tokenizer)

        if self.opt.do_predict is True:
            self.predictset = DatasetSQuAD2(opt.dataset_file['predict'], tokenizer, 'entity', self.opt.label_list)

        # build saver
        self.saver = tf.train.Saver(max_to_keep=1)
>>>>>>> 6b945f153e4c06b55dc990fab32bcf6ef7a49337
        
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
                context = sample_batched('text')
                query = sample_batched('query')
                answer = sample_batched('answer')

                model = self.model
                _ = self.session.run(model.train_op, feed_dict = {model.context:context, model.query:query, model.answer:answer})
                self.model = model

            except tf.errors.OutOfRangeError:
                break
        
         # = self._eval_metrics(val_data_loader)

    def _eval_metrics(self, data_loader):
        pass
        # iterator = data_loader.make_one_shot_iterator()
        # one_element = iterator.get_next()
        #
        # all_outputs = []
        # while True:
        #     try:
        #         sample_batched = self.session.run(one_element)
        #         inputs = sample_batched('text')
        #         #inputs = sample_batched('text')
        #         #inputs = sample_batched('text')
        #
        #         model = self.model
        #         outputs = self.session.run(model.output_op, feed_dict={})
        #
        #         all_outputs.extend(outputs)
        #     except tf.errors.OutOfRangeError:
        #         break
        #
        # metrics_1 =
        # logger.info(metrics_1)
        #
        # return metrics_1

        
    def run(self):

        # train and save best model
        best_model_path = self._train(None, self.opt.optimizer, self.trainset, self.testset)

        # calculate metric on test set 
        self.saver.restore(self.session, best_model_path)
        metric_1 = self._eval_metrics(self.test_data_loader)
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
            'train':'corpus/dureader/raw/trainset/zhidao.train.json',
            'test':'corpus/dureader/raw/testset/zhidao.test.json'},
        'marco':{
            'train':'',
            'test':''},
        'deepmin':{
            'train':'',
            'test':}
    }

    model_classes = {
        'bidaf': BIDAF,
        # 'mlstm': MLSTM,
        # 'bert': BERT,
        # 'bert_bidaf' : BERT_BIDAF
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



