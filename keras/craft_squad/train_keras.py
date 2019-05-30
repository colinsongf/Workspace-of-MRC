# -*- coding:utf-8 -*-
# author : apollo2mars@gmail.com

"""
train model by keras
data from DataHelper.py
args from keras_run.py

current model will rewrite last model, I will fixed later
There will be options to control whether rewrite the last model file
"""

import tensorflow as tf

from keras.models import Model  # need cudnn
from keras.layers import Input, LSTM, GRU, Embedding, Dense, Dropout, Concatenate, Average, Activation, Add, Multiply, Lambda, Flatten
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from keras import regularizers
from keras.optimizers import Adam
from keras import backend as K
from craft_squad.models_self.weightLayer import TrainableWeight
from utils.matrix_function import MatrixFunction
from craft_squad.models_self.predictLayer import PredictLayer

"""
Lambda function for train_tm
+ 为何要使用Lambda : 要封装 K.backend 函数
+ 为何要把Lambda 放到最前边 https://github.com/keras-team/keras/issues/8226
"""
f_reverse = Lambda(lambda x: K.reverse(x, 1))
f_repeat_128 = Lambda(lambda x: K.repeat(x, 128))
f_repeat_300 = Lambda(lambda x: K.repeat(x, 300))
f_exp = Lambda(lambda x: K.exp(x), name="exp")
f_transpose = Lambda(lambda x: K.permute_dimensions(x, pattern=(0, 2, 1)))
f_batch_dot = Lambda(lambda x: K.batch_dot(x[0], x[1]))
f_concatenate = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=2))  # K.concatenate(context_emb, f_align, axis=-1)
MAX_SPAN = 15


def find_best_answer(beg_dist, end_dist):
    sess = tf.Session()
    beg_list = sess.run(beg_dist)
    end_list = sess.run(end_dist)

    tmp_max = 0
    max_b_idx = 0
    max_e_idx = 0

    for idx, val_beg in enumerate(beg_list):
        for span in range(MAX_SPAN):
            tmp = idx + span
            if tmp < len(end_list):
                val_end = end_list[tmp]
            else:
                val_end = end_list[-1]

            value = val_beg * val_end
            if value > tmp_max:
                tmp_max = value
                max_b_idx = idx
                max_e_idx = idx + span
    return max_b_idx, max_e_idx


class QATrain():
    def __init__(self, data_inuse, args):
        """
        :param data_inuse: parameters from dataHelper
        :param args: parameters from argumentParser
        """
        self.embedding_matrix = data_inuse.embedding_matrix
        self.vocab_size = data_inuse.vocab_size
        self.pad_txt_train_cnt = data_inuse.pad_txt_train_cnt
        self.pad_txt_train_qry = data_inuse.pad_txt_train_qry
        self.pad_txt_dev_cnt = data_inuse.pad_txt_dev_cnt
        self.pad_txt_dev_qry = data_inuse.pad_txt_dev_qry
        self.idx_train_beg = data_inuse.idx_train_beg
        self.idx_train_end = data_inuse.idx_train_end
        self.idx_dev_beg = data_inuse.idx_dev_beg
        self.idx_dev_end = data_inuse.idx_dev_end

        self.log_folder_path = args.log_folder_path
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.epoch_nums = args.epoch_nums

        self.len_cnt_padding = args.context_padding_length
        self.len_qry_padding = args.query_padding_length
        self.encoder_unit_number = args.encoder_unit_number
        self.emb_out_size = args.embedding_layer_output_size
        self.emb_trainable = args.embedding_trainable

        self.dense_unit_number = args.dense_unit_number

        self.dropout = args.dropout

        self.reg_kernel = args.reg_kernel
        self.reg_activity = args.reg_activity

        self.learning_rate = args.learning_rate

        self.es_patience = args.es_patience

    def train＿baseline_DMRC(self):
        pass

    def train_tm(self):
        """
        word embedding
        """
        input_context = Input(shape=(self.len_cnt_padding,),
                              dtype='int32',
                              name='input_context')
        input_query = Input(shape=(self.len_qry_padding,),
                            dtype='int32',
                            name='input_query')

        """
        use same embedding layer to represent context embedding and query embedding
        """
        layer_emb = Embedding(input_dim=self.vocab_size,
                              output_dim=self.emb_out_size,
                              trainable=self.emb_trainable)

        context_emb = layer_emb(input_context)
        query_emb = layer_emb(input_query)
        """
        encode the context : reture the sequence of BiLSTM 
        """
        layer_context_encoder = Bidirectional(LSTM(units=self.encoder_unit_number // 2,
                                                   return_sequences=True,
                                                   kernel_regularizer=regularizers.l2(self.reg_kernel),
                                                   activity_regularizer=regularizers.l2(self.reg_activity)))
        """
        encode the query : concatenate the last hidden state vector of 
        """
        layer_query_encoder = LSTM(units=self.encoder_unit_number // 2,
                                   kernel_regularizer=regularizers.l2(self.reg_kernel),
                                   activity_regularizer=regularizers.l2(self.reg_activity))
        query_encode_forward = layer_query_encoder(query_emb)

        tmp_1 = f_reverse(query_emb)
        query_encode_backward = layer_query_encoder(tmp_1)

        u_vector = Concatenate(axis=1)([query_encode_forward, query_encode_backward])
        print("u_vector")
        print(K.print_tensor(u_vector))  # (:, 128)

        u_repeat_300 = f_repeat_300(u_vector)

        print("u_300")
        print(K.print_tensor(u_repeat_300))  # (:, 300, 128)

        """
        context represnet
        """
        context_encode_sequence = layer_context_encoder(context_emb)  # shape (:, len_cnt_padding, encoder_unit_number)
        """
        get attention s(t)
        """
        w_yd = TrainableWeight(self.encoder_unit_number, self.encoder_unit_number, self.reg_kernel)
        w_u = TrainableWeight(self.encoder_unit_number, self.encoder_unit_number, self.reg_kernel)

        weight_context = w_yd(context_encode_sequence)  # (:,300,128) (128, 128)
        weight_query = w_u(u_repeat_300)  # (:,300,128) (128, 128)

        weight_add = f_exp(Add(name="add")([weight_context, weight_query]))  # (:,300,128)
        weight_tanh = Activation('tanh', name="tanh")(weight_add)  # (:,300,128)

        # next weight
        weight_exp = TrainableWeight(self.encoder_unit_number, 1, self.reg_kernel)  # (128, 1)
        s = weight_exp(weight_tanh)  # (:, 300, 1)

        context_encode_sequence_transpose = f_transpose(context_encode_sequence)  # (:,128,300)
        print("context_encode_sequence_transpose")
        print(K.print_tensor(context_encode_sequence_transpose))

        # get represent of attention context

        r = f_batch_dot([context_encode_sequence_transpose, s])  # (:,128, 300)  (:,300, 1)
        print("r")
        print(K.print_tensor(r))  # (:, 128, 1)

        """
        get all represent of context and query
        """
        # w_4 = TrainableWeight(self.encoder_unit_number, 1, self.reg_kernel)
        # w_5 = TrainableWeight(self.encoder_unit_number, 1, self.reg_kernel)
        # context_encode = Multiply()([Flatten()(w_4), Flatten()(r)])  # 128 * 128
        # query_encode = Multiply()([Flatten()(w_5), Flatten()(u_2)])  # 128 * 128
        # print("context_encode")
        # print(K.print_tensor(context_encode))  # (:, 128, 1)
        # print("query_encode")
        # print(K.print_tensor(query_encode))  # (:, 128, 1)

        # add context and query represent
        tmp_add = Add()([r, u_vector])
        print("tmp_add")
        print(K.print_tensor(tmp_add))
        concate_context_query = Activation('tanh')(tmp_add)
        print("concate_context_query")
        print(K.print_tensor(concate_context_query))

        """
        handle the context and query match
        """
        # concate_context_query = K.concatenate([context_encode, query_encode], axis=1)
        # # concate_context_query = Concatenate(axis=-1)[context_encode, query_encode]

        # repre_context_query = LSTM(units=self.encoder_unit_number)(concate_context_query)

        """
        fully connected and softmax layer: beging and end index of the answer

        todo : nce, hierarchical softmax, negative sampling
        """

        concate_context_query_flatten = Flatten()(concate_context_query)
        print("concate_context_query_flatten")
        print(K.print_tensor(concate_context_query_flatten))

        dropout_beg = Dropout(self.dropout)(concate_context_query_flatten)
        idx_beg_tmp = Dense(self.len_cnt_padding,
                            kernel_regularizer=regularizers.l2(self.reg_kernel),
                            activity_regularizer=regularizers.l2(self.reg_activity),
                            activation='sigmoid',
                            name="idx_beg_tmp")(dropout_beg)
        beg = Dense(self.len_cnt_padding, activation='sigmoid',
                    name='idx_beg_out')(idx_beg_tmp)

        dropout_end = Dropout(self.dropout)(concate_context_query_flatten)
        idx_end_tmp = Dense(self.len_cnt_padding,
                            kernel_regularizer=regularizers.l2(self.reg_kernel),
                            activity_regularizer=regularizers.l2(self.reg_activity),
                            activation='sigmoid',
                            name='idx_end_tmp')(dropout_end)
        end = Dense(self.len_cnt_padding, activation='sigmoid',
                    name='idx_end_out')(idx_end_tmp)
        """
        define the input and output of the model
        """
        model = Model(inputs=[input_context, input_query],
                      outputs=[beg, end])
        model.summary()
        """
        model compile
        """
        opt = Adam(lr=0.0001,
                   beta_1=0.8,
                   beta_2=0.999,
                   epsilon=1e-09)
        # metrics =

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        # metrics=['sparse_categorical_accuracy'])  # https://keras.io/zh/metrics/
        """
        tensorboard setting
        """
        # tb_cb = TensorBoard(log_dir=self.log_folder_path,
        #                     write_graph=True,
        #                     write_images=1,
        #                     histogram_freq=1)
        # cbks = [tb_cb]
        """
        model fit
        """
        model.fit(x=[self.pad_txt_train_cnt, self.pad_txt_train_qry],
                  y=[self.idx_train_beg, self.idx_train_end],
                  epochs=self.epoch_nums,
                  batch_size=self.batch_size,
                  verbose=1,
                  validation_data=[[self.pad_txt_dev_cnt, self.pad_txt_dev_qry], [self.idx_dev_beg, self.idx_dev_end]],
                  # validation_split=0.05,
                  # callbacks=cbks
                  )
        """
        model save
        """
        model.save(self.model_name)

    def train_AoA(self):
        pass
    #     """
    #     word embedding
    #     """
    #     input_context = Input(shape=(self.len_cnt_padding,),
    #                           dtype='int32',
    #                           name='input_context')
    #     input_query = Input(shape=(self.len_qry_padding,),
    #                         dtype='int32',
    #                         name='input_query')
    #     # use same embedding layer to represent context embedding and query embedding
    #     layer_emb = Embedding(input_dim=self.vocab_size,
    #                           output_dim=self.emb_out_size,
    #                           trainable=True)
    #
    #     context_emb = layer_emb(input_context)
    #     query_emb = layer_emb(input_query)
    #
    #     """
    #     encode the context : reture the sequence of BiLSTM
    #     """
    #     layer_context_encoder = Bidirectional(LSTM(units=self.encoder_unit_number // 2,
    #                                                return_sequences=True,
    #                                                kernel_regularizer=regularizers.l2(self.reg_kernel),
    #                                                activity_regularizer=regularizers.l2(self.reg_activity)))
    #
    #     context_encode_sequence = layer_context_encoder(context_emb)  # shape (:, len_cnt_padding, encoder_unit_number)
    #
    #     """
    #     encode the query : concatenate the last hidden state vector of
    #     """
    #     layer_query_encoder = Bidirectional(LSTM(units=self.encoder_unit_number // 2,
    #                                              return_sequences=True,
    #                                              kernel_regularizer=regularizers.l2(self.reg_kernel),
    #                                              activity_regularizer=regularizers.l2(self.reg_activity)))
    #     query_encode_sequence = layer_query_encoder(query_emb)
    #
    #     """
    #     get M : when i th word of document and j th word of query, we can compute a matching score by their dot product
    #     """
    #     context_encode_sequence_tranpose = keras.backend.permute_dimensions(context_encode_sequence, (0,2,1))
    #     M_Matrix = K.dot(permute_dimensions(context_encode_sequence_tranpose, query_encode_sequence))
    #
    #     # column softmax
    #     new_Matrix = get_col_softmax(M_Matrix)
    #     # column wise average
    #     col_avg = get_col_average(M_Matrix)
    #
    #     ###
    #     How to predict Answer
    #     ###

    def train＿baseline_SQuAD(self):
        pass

    def train_bidaf(self):
        pass

    def train_rnet(self):
        pass

    def train_DrQA(self):
        """
        word embedding
        """
        input_context = Input(shape=(self.len_cnt_padding,),
                              dtype='float32',
                              name='input_context')
        input_query = Input(shape=(self.len_qry_padding,),
                            dtype='float32',
                            name='input_query')

        """
        use same embedding layer to represent context embedding and query embedding
        """
        layer_emb = Embedding(input_dim=self.vocab_size,
                              output_dim=self.emb_out_size,
                              weights=[self.embedding_matrix],
                              trainable=self.emb_trainable)

        # (:, 350, 300)
        context_emb = layer_emb(input_context)
        # (:, 30, 300)
        query_emb = layer_emb(input_query)
        """
        get attention s(t)
        """
        transpose_query_emb = f_transpose(query_emb)  # (:, 300, 30)
        # (:, 300, 30)
        att_matrix = f_batch_dot([context_emb, transpose_query_emb])  # (:, 350, 300) * transpose((:, 30, 300)) ==> (:,350, 30)
        mf = MatrixFunction()
        row_softmax = mf.softmax_row_tensor(att_matrix) # (:, 350, 30)
        transpose_row_softmax = f_transpose(row_softmax)  # (:, 30, 350)

        """
        query2context
        f_align(pi) = \sum{ a_{ij} E(q_j)}
        (:, 300, 30) * (:, 30, 350) ==> (:, 300, 350)
        """
        f_align_tmp = f_batch_dot([transpose_query_emb, transpose_row_softmax])  # (:, 300, 350)
        f_align = f_transpose(f_align_tmp)  # (:, 350, 300)


        """
        get all p_i
        f_emb         + f_align      
        (:, 350, 3000) + (:, 350, 300) ==> (:, 350, 600)
        
        other information should be added in the futures
        """
        context_encoding_sequence = f_concatenate([context_emb, f_align])
        context_encoding_sequence = Dropout(self.dropout)(context_encoding_sequence)


        """
        Question Encoding : Add another recurrent neural network on top of of the word embeddings of q_i
        and combine the resulting hidden units into one single vector : {q_1, ..., q_l} -> q
        """
        # (:, 128) question 的向量表示

        layer_gru = GRU(units=self.emb_out_size,
                        kernel_regularizer=regularizers.l2(self.reg_kernel),
                        activity_regularizer=regularizers.l2(self.reg_activity)
                        )
        query_encoding_single_vector = layer_gru(query_emb)  # (:, 300)

        """
        Prediction : At the paragraph level, the goal is to predict the span of tokens that is most likely the right answer
        """
        layer_begin = PredictLayer(self.len_cnt_padding, self.emb_out_size * 2, self.emb_out_size, self.reg_kernel)
        begin_distribution = layer_begin([context_encoding_sequence, query_encoding_single_vector])  # (:, 350)
        # begin_distribution = Dropout(self.dropout)(begin_distribution)
        begin_distribution = Dense(self.len_cnt_padding*2, activation='relu')(begin_distribution)
        # begin_distribution = Dropout(self.dropout)(begin_distribution)
        begin_distribution = Dense(self.len_cnt_padding, activation='softmax', name='begin')(begin_distribution)

        layer_end = PredictLayer(self.len_cnt_padding, self.emb_out_size * 2, self.emb_out_size, self.reg_kernel)
        end_distribution = layer_end([context_encoding_sequence, query_encoding_single_vector])  # (:, 350)
        # end_distribution = Dropout(self.dropout)(end_distribution)
        end_distribution = Dense(self.len_cnt_padding*2, activation='relu')(end_distribution)
        # end_distribution = Dropout(self.dropout)(end_distribution)
        end_distribution = Dense(self.len_cnt_padding, activation='softmax', name="end")(end_distribution)

        """
        During prediction:
            + we choose the best span from token i to token i^' such as i< i^'< i+15
            + and P_start(i) * P_end(i^') is maximized 
        """
        # begin_index, end_index = find_best_answer(begin_distribution, end_distribution)

        """
        define the input and output of the model
        """
        model = Model(inputs=[input_context, input_query], outputs=[begin_distribution, end_distribution])
        model.summary()
        """
        model compile
        """
        opt = Adam(lr=self.learning_rate, beta_1=0.8, beta_2=0.999, epsilon=1e-09)
        # metrics =

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        # metrics=['sparse_categorical_accuracy'])  # https://keras.io/zh/metrics/
        """
        tensorboard setting
        """
        # tb_cb = TensorBoard(log_dir=self.log_folder_path,
        #                     write_graph=True,
        #                     write_images=1,
        #                     histogram_freq=1)

        """
        early stopping
        """
        es = EarlyStopping(monitor='val_loss', patience=self.es_patience, verbose=0, mode='auto')

        cbks = [es]

        """
        model fit
        """
        model.fit(x=[self.pad_txt_train_cnt, self.pad_txt_train_qry],
                  y=[self.idx_train_beg, self.idx_train_end],
                  epochs=self.epoch_nums,
                  batch_size=self.batch_size,
                  verbose=1,
                  validation_data=[[self.pad_txt_dev_cnt, self.pad_txt_dev_qry], [self.idx_dev_beg, self.idx_dev_end]],
                  # validation_split=0.05,
                  callbacks=cbks
                  )
        """
        model save
        """
        model.save(self.model_name)



