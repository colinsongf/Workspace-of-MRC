# -*- utf-8 -*-
# author : apollo2mars@gmail.com

"""
The main entry of solution keras
contains:
    the setting of the model
    the setting of what to do, "train" or "test" (current just test on dev)
"""

import argparse
from data_helper import DataHelper
from train_keras import QATrain
from test_keras import QATest

TASK_TYPE = 'test'
BATCH_SIZE = 32
REGULARIZATION_KERNEL = 3e-4
REGULARIZATION_ACTIVITY = 3e-3
EPOCH_NUMBER = 10000
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10
DROPOUT = 0.2
FRAMEWORK_CHANGE = "ADD A DENSE of 600"
MAX_SPAN = 15

parameters_list = [str(REGULARIZATION_KERNEL), str(REGULARIZATION_ACTIVITY),
                   str(EPOCH_NUMBER), str(LEARNING_RATE), str(EARLY_STOPPING_PATIENCE), str(DROPOUT), FRAMEWORK_CHANGE]
parameters_str = "##".join(parameters_list)

MODEL_NAME = "SQuADv1.1#" + parameters_str

RESULT_NAME = "SQuADv1.1_predict_results.txt"


def setup_args():
    """
    argparse
    """
    parser = argparse.ArgumentParser(description="Make some settings")
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument("--task_type", type=str,
                        default=TASK_TYPE,
                        help="task type contain train/test/all/check")
    """
    batch setting
    """
    parser.add_argument("--batch_size",
                        type=int, nargs="?", const=True,
                        default=BATCH_SIZE,
                        help="batch size of train and test")
    """
    embedding settings
    """
    parser.add_argument("--embedding_layer_output_size",
                        type=int, nargs="?", const=True,
                        default=300,
                        help="size of embedding layer output")

    parser.add_argument("--embedding_trainable",
                        default=True)

    """
    tuning parameters
    """
    parser.add_argument("--context_padding_length", type=int, nargs="?", const=True,
                        default=350, help="length of context padding")
    parser.add_argument("--query_padding_length", type=int, nargs="?", const=True,
                        default=30, help="length of query padding")
    parser.add_argument("--encoder_unit_number", type=int, nargs="?", const=True,
                        default=128, help="unit number of encoder")
    parser.add_argument("--dense_unit_number", type=int, nargs="?", const=True,
                        default=301, help="unit number of fully connected layer")
    """
    regularization　parameters
    """
    parser.add_argument("--reg_kernel", type=float, nargs="?", const=True,
                        default=REGULARIZATION_KERNEL, help="regularization of kernel, LSTM, Dense, PredictLayer")
    parser.add_argument("--reg_activity", type=float, nargs="?", const=True,
                        default=REGULARIZATION_ACTIVITY, help="regularization of activity, LSTM, Dense")

    """
    learning parameters
    """
    parser.add_argument("--epoch_nums", type=int, nargs="?", const=True,
                        default=EPOCH_NUMBER, help="number of training epochs, if epoch number is 0  means unlimited training")

    parser.add_argument("--learning_rate", type=float, nargs="?", const=True,
                        default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--es_patience", type=int, nargs="?", const=True,
                        default=EARLY_STOPPING_PATIENCE, help="early stopping patience")
    parser.add_argument("--dropout", type=float, nargs="?", const=True,
                        default=DROPOUT, help="dropout : fraction of the entries in the tensor that will be set to 0")

    """
    file settings parameters
    """
    parser.add_argument("--model_name", type=str, nargs="?", const=True,
                        default=MODEL_NAME, help="model name for train and test")
    parser.add_argument("--log_folder_path", type=str, nargs="?", const=True, default="/tmp/squad_tesorboard/", help="the folder name of tensorboard log path")
    parser.add_argument("--RESULT_NAME", type=str, nargs="?", const=True,
                        default=RESULT_NAME, help="model name for train and test")

    """
    test setting
    """
    parser.add_argument("--MAX_SPAN", type=int, nargs="?", const=True,
                        default=MAX_SPAN, help="the max span of begin and end index")

    return parser.parse_args()


if __name__ == "__main__":
    # get some args of this solution
    args = setup_args()
    # get train and dev(test) data
    data_inuse = DataHelper(args)
    data_inuse.run()  # update values of the attribute of DataHelper

    if args.task_type == 'train':
        K_train = QATrain(data_inuse, args)
        # K_train.train_baseline()
        K_train.train_DrQA()
    elif args.task_type == 'test':
        K_test = QATest(data_inuse, args)
        K_test.test_DrQA()
        K_test.check_results()
    elif args.task_type == 'all':
        K_train = QATrain(data_inuse, args)
        # K_train.train_baseline()
        K_train.train_DrQA()

        K_test = QATest(data_inuse, args)
        K_test.test_DrQA()
        K_test.check_results()

    elif args.task_type == 'check':
        K_test = QATest(data_inuse, args)
        K_test.baseline_check()
    else:
        print("input task is error, please check your input")
