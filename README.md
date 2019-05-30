[TOC]
# Dataset

## DuReader Dataset
DuReader is a new large-scale real-world and human sourced MRC dataset in Chinese. DuReader focuses on real-world open-domain question answering. The advantages of DuReader over existing datasets are concluded as follows:
 - Real question
 - Real article
 - Real answer
 - Real application scenario
 - Rich annotation

## SQuAD Dataset

+ links

##  MARCO V2

+ [MS MARCO](http://www.msmarco.org/dataset.aspx) (Microsoft Machine Reading Comprehension) is an English dataset focused on machine reading comprehension and question answering. The design of MS MARCO and DuReader is similar. It is worthwhile examining the MRC systems on both Chinese (DuReader) and English (MS MARCO) datasets. 

+ You can download MS MARCO V2 data, and run the following scripts to convert the data from MS MARCO V2 format to DuReader format. Then, you can run and evaluate our DuReader baselines or your DuReader systems on MS MARCO data. 




# Process

## Download the Dataset

To Download DuReader dataset:

```
cd data && bash download.sh
```

For more details about DuReader dataset please refer to [DuReader Homepage](https://ai.baidu.com//broad/subordinate?dataset=dureader).

## Download Thirdparty Dependencies

We use Bleu and Rouge as evaluation metrics, the calculation of these metrics relies on the scoring scripts under "https://github.com/tylin/coco-caption", to download them, run:

```
cd utils && bash download_thirdparty.sh
```

## Preprocess the Data

After the dataset is downloaded, there is still some work to do to run the baseline systems. DuReader dataset offers rich amount of documents for every user question, the documents are too long for popular RC models to cope with. In our baseline models, we preprocess the train set and development set data by selecting the paragraph that is most related to the answer string, while for inferring(no available golden answer), we select the paragraph that is most related to the question string. The preprocessing strategy is implemented in `utils/preprocess.py`. To preprocess the raw data, you should first segment 'question', 'title', 'paragraphs' and then store the segemented result into 'segmented_question', 'segmented_title', 'segmented_paragraphs' like the downloaded preprocessed data, then run:

```
cat data/raw/trainset/search.train.json | python utils/preprocess.py > data/preprocessed/trainset/search.train.json
```

The preprocessed data can be automatically downloaded by `data/download.sh`, and is stored in `data/preprocessed`, the raw data before preprocessing is under `data/raw`.




# Framework

## Run Tensorflow

We also implements the BIDAF and Match-LSTM models based on Tensorflow 1.0. You can refer to the [official guide](https://www.tensorflow.org/versions/r1.0/install/) for the installation of Tensorflow. The complete options for running our Tensorflow program can be accessed by using `python run.py -h`. Here we demonstrate a typical workflow as follows: 

### Preparation

Before training the model, we have to make sure that the data is ready. For preparation, we will check the data files, make directories and extract a vocabulary for later use. You can run the following command to do this with a specified task name:

```
python run.py --prepare
```
You can specify the files for train/dev/test by setting the `train_files`/`dev_files`/`test_files`. By default, we use the data in `data/demo/`

### Training

To train the reading comprehension model, you can specify the model type by using `--algo [BIDAF|MLSTM]` and you can also set the hyper-parameters such as the learning rate by using `--learning_rate NUM`. For example, to train a BIDAF model for 10 epochs, you can run:

```
python run.py --train --algo BIDAF --epochs 10
```

The training process includes an evaluation on the dev set after each training epoch. By default, the model with the least Bleu-4 score on the dev set will be saved.

### Evaluation

To conduct a single evaluation on the dev set with the the model already trained, you can run the following command:

```
python run.py --evaluate --algo BIDAF
```

### Prediction

You can also predict answers for the samples in some files using the following command:

```
python run.py --predict --algo BIDAF --test_files ../data/demo/devset/search.dev.json 
```

By default, the results are saved at `../data/results/` folder. You can change this by specifying `--result_dir DIR_PATH`.



## Run PyTorch

## Run PaddlePaddle

## Run Keras

