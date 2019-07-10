from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from keras import regularizers


class PredictLayer(Layer):
    def __init__(self, len_cnt, len_cnt_emb, len_qn_emb, reg_kernel, **kwargs):
        self.kernel = None
        self.len_cnt = len_cnt
        self.len_cnt_emb = len_cnt_emb
        self.len_qn_emb = len_qn_emb
        self.reg_kernel = reg_kernel
        super(PredictLayer, self).__init__(**kwargs)

    # define weight
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      dtype='float32',
                                      shape=(self.len_cnt_emb, self.len_qn_emb),
                                      initializer=RandomNormal(mean=0.0, stddev=0.5, seed=None),
                                      regularizer=regularizers.l2(self.reg_kernel),
                                      trainable=True)
        super(PredictLayer, self).build(input_shape)  # Be sure to call this somewhere!

    # 这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量
    def call(self, x):
        all_cnt = x[0]  # context_padding_length, dim_lstm_context
        all_qst = x[1]  # question_padding_length, dim_lstm_question
        tmp1 = K.dot(all_cnt, self.kernel)
        tmp2 = K.batch_dot(tmp1, all_qst)
        tmp3 = K.exp(tmp2)
        return tmp3

    # 如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.len_cnt)

    def get_config(self):
        config = {'len_cnt': self.len_cnt,
                  'len_cnt_emb': self.len_cnt_emb,
                  'len_qn_emb': self.len_qn_emb,
                  'reg_kernel': self.reg_kernel}
        base_config = super(PredictLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))