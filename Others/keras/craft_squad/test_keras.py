# -*- utf-8 -*-
# author : apollo2mars@gmail.com

"""
Just test on dataset

"""

from keras.models import load_model
import pickle

from solution_keras_SQuAD.models_self.weightLayer import TrainableWeight
from solution_keras_SQuAD.models_self.predictLayer import PredictLayer


def handle_result(results, MAX_SPAN):
    """
    check the result of model predict
    """
    all_begin_probability_distribution = results[0]
    all_end_probability_distribution = results[1]

    all_begin_index_list = []
    all_end_index_list = []

    # 在一组 起始坐标 和终止坐标的分布(都是350 维)中, 找到在满足约束的情况下, 概率最大的span
    # context is : every may since 1987
    # answer is : may
    # span is : 1 1
    for tmp_bpd, tmp_epd in zip(all_begin_probability_distribution, all_end_probability_distribution):
        list_beg = list(tmp_bpd)  # (350)
        list_end = list(tmp_epd)  # (350)

        idx = 0

        max_b_idx = 0
        max_e_idx = 0
        tmp_max = 0

        while idx < len(list_beg):
            v1 = list_beg[idx]
            for span in range(MAX_SPAN):
                tmp = idx+span
                if tmp < len(list_beg):
                    v2 = list_end[tmp]
                else:
                    v2 = list_end[-1]

                value = v1 * v2
                if value > tmp_max:
                    tmp_max = value
                    max_b_idx = idx
                    max_e_idx = idx + span
            idx += 1
        all_begin_index_list.append(max_b_idx)
        all_end_index_list.append(max_e_idx)
            
    assert len(all_begin_index_list) == len(all_end_index_list)

    tmp_str_list = list(map(lambda x, y: str(x) + '\t' + str(y) + '\n', all_begin_index_list, all_end_index_list))
    return tmp_str_list


def do_check(contexts, results, right_idx_beg_list, right_idx_end_list):
    """
    check the result of model predict
    """
    assert len(results) == len(right_idx_beg_list)
    assert len(results) == len(right_idx_end_list)
    assert len(results) == len(contexts)

    count_EM = 0
    count_F1 = 0.0

    for context, answer_idx, right_idx_beg, right_idx_end in zip(contexts, results, right_idx_beg_list, right_idx_end_list):
        tmp = answer_idx.strip().split('\t')
        predict_beg = int(tmp[0])
        predict_end = int(tmp[1])

        predict_answer = context[predict_beg:predict_end+1]  # +1 is needed
        right_answer = context[right_idx_beg:right_idx_end+1]

        set_pred = set(predict_answer)
        set_right = set(right_answer)

        if len(set_pred - set_right) is 0:
            count_EM += 1
            count_F1 += 1
        else:
            precision = len(list(set_pred & set_right)) / len(set_right)
            recall = len(list(set_pred & set_right)) / len(set_pred)

            tmp = precision + recall

            if tmp == 0.0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

            count_F1 += f1

    return count_EM/len(results), count_F1/len(results)


class QATest():
    def __init__(self, data_test, args):
        self.model_name = args.model_name

        self.pad_txt_dev_cnt = data_test.pad_txt_dev_cnt
        self.pad_txt_dev_qry = data_test.pad_txt_dev_qry
        self.idx_dev_beg = data_test.idx_dev_beg
        self.idx_dev_end = data_test.idx_dev_end

        self.MAX_SPAN = args.MAX_SPAN
        self.RESULT_NAME = args.RESULT_NAME

        # 使用训练集进行测试
        # self.pad_txt_dev_cnt = data_test.pad_txt_train_cnt
        # self.pad_txt_dev_qry = data_test.pad_txt_train_qry
        # self.idx_dev_beg = data_test.idx_train_beg
        # self.idx_dev_end = data_test.idx_train_end

    def test_baseline(self):
        """
        custom_objects is needed for self-define layers
        """
        model = load_model(self.model_name, custom_objects={'TrainableWeight': TrainableWeight})

        results_probability_distribution = model.predict(x=[self.pad_txt_dev_cnt, self.pad_txt_dev_qry],
                                                         batch_size=512, verbose=0)
        results_index = handle_result(results_probability_distribution, self.MAX_SPAN)

        """
        save the result
        """
        with open("beg_and_end_index_result.txt", 'wb') as file_results:
            pickle.dump(results_index, file_results, 0)

    def test_DrQA(self):
        """
        custom_objects is needed for self-define layers
        """
        model = load_model(self.model_name, custom_objects={'PredictLayer': PredictLayer})

        results_probability_distribution = model.predict(x=[self.pad_txt_dev_cnt, self.pad_txt_dev_qry],
                                                         batch_size=64, verbose=0)

        results_index = handle_result(results_probability_distribution, self.MAX_SPAN)

        """
        save the result
        """
        with open(self.RESULT_NAME, 'wb') as file_results:
            pickle.dump(results_index, file_results, 0)

    def check_results(self):
        with open(self.RESULT_NAME, 'rb') as file_results:
            results_index = pickle.load(file_results)
        em, f1 = do_check(self.pad_txt_dev_cnt, results_index, self.idx_dev_beg, self.idx_dev_end)
        print("EM is {}".format(em))
        print("F1 is {}".format(f1))
