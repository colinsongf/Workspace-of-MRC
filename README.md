[TOC]

# Dataset

| Dataset                                                      | SOTA                                                         | Tips |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)     | BERT + DAE + AoA (ensemble) Joint Laboratory of HIT and iFLYTEK Research |      |
| [MARCO V2](http://www.msmarco.org/leaders.aspx)              | **Enriched BERT base + AOA index V1** Ming Yan of Alibaba Damo NLP |      |
| [DuReader](http://ai.baidu.com/broad/subordinate?dataset=dureader) |                                                              |      |
| LAMBADA                                                      |                                                              |      |
| Who-did-What(WDW)                                            |                                                              |      |
| CNN & DailyMail(close style)                                 |                                                              |      |
| Children's Book Test BookTest                                |                                                              |      |



# Metric

- MAP
- MRR
  - 是一个国际上通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，如果没有匹配的句子分数为0
  - 最终的分数为所有得分之和
- NDCG
- EM
- F1
- ROUGE-L
- BLEU

# Tasks and Solutions

| Tasks       |      |      |
| ----------- | ---- | ---- |
| Factoid     |      |      |
| Close Style |      |      |
| Open domain |      |      |



| Solutions            |      |      |
| -------------------- | ---- | ---- |
| Search and Retrieval |      |      |
| Knowledge based      |      |      |
| Semantic Parsing     |      |      |
| Language Model       |      |      |
| Generative           |      |      |



# Paper List

## BIDAF

## Matching-LSTM

## Hybird Attention-over-Attention Reader

## Reading Wikipedia to Answer Open Domain Questions

## Reference

- 2017年　以前的论文和数据集整理](https://www.zybuluo.com/ShawnNg/note/622592)
- [2018年　清华77篇机器阅读理解论文](http://www.zhuanzhi.ai/document/87418ceee95a21622d1d7a21f71a894a)
- 2019 年　有待整理
- [SQuAD 的一些模型](<http://www.zhuanzhi.ai/document/d6a0038bb0143a6805370adb58bf68be>)



#  Project Structure

## Process

### Download the Dataset

### Download Thirdparty Dependencies

### Preprocess the Data

## Run Tensorflow

### Preparation

### Training

### Evaluation

### Prediction

## Run PyTorch

## Run PaddlePaddle

## Run Keras

## Untils

### Mulit Card Training

# IDEA and Next

- Pretrain Embedding and Fine tuning
- Information Retrieval
  - For Open domain
    - web crawler 
    - wikipedia
    - konwledge base
    - SQL(NL2SQL)
  - For Task Specific
    - special dataset or corpus
- NLU
  - Emotion : Sentiment Analysis
  - Classification : Multi Category and Hierachical Level Category
  - Semantic Parsing
    - POS/TAG/NER/
  - Syntax Analysis
  - Chapter Analysis
  - Information Extraction
    - summarization 
- Global Logic(Language Modeling or Information Modeling)
  - Dynamic Embedding
  - Entity Relation
  - Knowledge Base
  - structure
    - graph 
    - matirx
- NLG
  - span
  - pointer
  - generation
    - reinforce learning for answer regeneration
    - answer generation : NLG
      - GAN
      - VAE 
- Multi-Task/Joint/Transfor Learning
  - optimizer
  - loss function
  - ranking
- Unsupervised MRC 

# Milestone

+ pass

# Reference

+ pass

