

### Environment
+ python3

### Prepare
1. install requirements.txt
2. download embedding data and save in folder 'data'(to be changed)
3. squad data is already in folder 'data', and those data have been processed

### Data
+ SQuAD 1.1
    + train data information
        + SQuAD 1.1 train data information
        + context average number of character is 123.22429519928848
        + context max number of character is 679
        + question average number of character is 10.249829358595155
        + question max number of character is 40
        + index of answer is index of word, not character
    + dev data information
    + test data information
        + Only the training and validation data are publicly available, while the test data is hidden that one has to submit the code to a Codalab and work with the authors to retrieve the final test score
        + https://rajpurkar.github.io/SQuAD-explorer/

+ embedding Glove http://nlp.stanford.edu/data/wordvecs/glove.6B.zip


### solution_keras
0. the code is developed by myself
1. the alforithm refers to the idea of DrQA(just Reader, not Retrieval)

###### Operation Guide for solution_keras
0. cd 'solution_keras'
1. run 'keras_run.py'

### solution_tensorflow
+ code is from https://github.com/priya-dwivedi/cs224n-Squad-Project (python2)
+ and fix some problem in order to run in python3
+ this solution contain BiDAF, R-NET, and the ensemble of these algorithms.

###### Operation Guide for solution_tensorflow
0. cd 'solution_tensorflow_SQuAD'
1. run 'tf_model_build.py'


###### Requirement
numpy==1.14.0
tensorflow-gpu==1.4.1  # Change to tensorflow==1.4.1 if you need to run CPU-only tensorflow (e.g. on your laptop)
tensorflow-tensorboard==0.4.0
Keras==2.1.2

There will be problems with other versions


##### Outline of Algorithm Architecture
+ char embedding
+ word embedding
+ concatenation of char embedding and word embedding
+ context and question represent
    + Plan A(Teach machine to read and comprehend)
        +
    + Plan B(DrQA)
        + just contain context information
            + LSTM
        + contain the match of context and query
            + context 2 query
            + query 2 context
        + full represent
            + context
                + LSTM(not sequence) + context2query
            + query
                + LSTM(not sequence) + query2context
    + Plan C(BiDAF)
    + Plan D(R-Net)
    + Plan E(QA-Net)
    + Plan F(Attention over Attention Neural Networks for Reading Comprehension)
+ handle the represent
+ answer
    + begin and end index predict
        + dense
        + softmax
        + loss the results
            + sparse cross entropy
            + self define loss