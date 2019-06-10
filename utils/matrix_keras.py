# -*- coding:utf-8 -*-
# some matrix function of vector, matrix(row, col)
# author : apollo2mars@gmail.com

import math
import tensorflow as tf
import keras.backend as K
from keras.layers import Lambda

f_softmax = Lambda(lambda x: K.softmax(x))


class MatrixFunction(object):
    def __init__(self):
        pass

    """
    input matrix is tf.Variable
    x: row index
    y: col index
    """
    def set_value(self, matrix, x, y, val):
        # 提取出要更新的行
        row = tf.gather(matrix, x)
        # 构造这行的新数据
        new_row = tf.concat([row[:y], [val], row[y + 1:]], axis=0)
        # 使用 tf.scatter_update 方法进正行替换
        matrix.assign(tf.scatter_update(matrix, x, new_row))
    """
    return the attention weight in single row
    input is tensor(dimension is > 2) 
    """
    def softmax_row_tensor(self, matrix):
        return f_softmax(matrix)

    def softmax_col(self, matrix):
        matrix_exp = tf.exp(matrix)
        matrix_shape = matrix_exp.shape
        if len(matrix_shape) < 2 or len(matrix_shape) > 3:
            print("input dimension is less than 2 or more 3")
        elif len(matrix_shape) is 2:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            # 转化为numpy数组
            # matrix_np = matrix.eval(session=sess)
            matrix_np = sess.run(matrix_exp)

            for j in range(matrix_shape[1]):  # 遍历所有的列
                matrix_np[:, j] = matrix_np[:, j] / sum(matrix_np[:, j])  # 求 softmax 值
            tmp_return = tf.convert_to_tensor(matrix_np)
            return tmp_return
        elif len(matrix_shape) is 3:
            tmp_init = tf.Variable(tf.zeros(matrix_shape))

            for i_0 in range(matrix_shape[0]):  # 遍历第一个维度, batch 维度
                tmp = tf.gather(matrix, [i_0], axis=0)
                tmp = tf.reshape(tmp, tmp.shape[1:])
                tmp_1 = self.softmax_col(tmp)
                tmp_init = tf.scatter_update(tmp_init, i_0, tmp_1)
            return tmp_init

    """
    input is a tensor variable
    """
    def average_col(self, matrix):
        matrix_shape = matrix.shape
        if len(matrix_shape) < 2 or len(matrix_shape) > 3:
            print("input dimension is less than 2 or more 3")
        elif len(matrix_shape) is 2:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            # 转化为numpy数组
            # matrix_np = matrix.eval(session=sess)
            matrix_np = sess.run(matrix)
            tmp_val = []
            for j in range(matrix_shape[1]):  # 遍历所有的列
                all_val = sum(matrix_np[:, j])  # 当前列求和
                tmp_val.append(all_val/int(matrix_shape[0]))  # 求 softmax 值
            tmp_return = tf.convert_to_tensor(tmp_val)
            return tmp_return
        elif len(matrix_shape) is 3:
            tmp_init = tf.Variable(tf.zeros([int(matrix_shape[0]), int(matrix_shape[-1])]))
            for i_0 in range(matrix_shape[0]):  # 遍历第一个维度, batch 维度
                tmp = tf.gather(matrix, [i_0], axis=0)  # (1, 4, 3)
                tmp = tf.reshape(tmp, tmp.shape[1:])  # (4, 3)
                tmp_1 = self.average_col(tmp)  # (3)
                tmp_init = tf.scatter_update(tmp_init, i_0, tmp_1)
            return tmp_init


if __name__ == "__main__":
    tensor1 = tf.Variable([[[1,1,1], [1,3,3], [1,6,4], [1,6,8]],
                           [[3,3,1], [3,3,1], [4,4,1], [4,4,1]],
                           [[5,5,1], [5,5,1], [6,6,1], [6,6,1]],
                          ], dtype=tf.float32)
    mf = MatrixFunction()
    tensor2 = mf.softmax_row(tensor1)
    tensor3 = mf.softmax_col(tensor1)
    tensor4 = mf.average_col(tensor1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tensor1))
        print("\n\n row softmax \n\n")
        print(sess.run(tensor2))
        print("\n\n col softmax \n\n")
        print(sess.run(tensor3))
        print("\n\n col average \n\n")
        print(sess.run(tensor4))


