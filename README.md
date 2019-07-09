[TOC]

# Workspace of Machine Reading Comprehension

# Target

+ Algorithms implementation of **M**achine **R**eading **C**omprehension 
+ Efficient and beautiful code
+ General Architecture for MRC
    + Framework for multi-dataset and multi-card
    + Framework for model ensemble

# Dataset

| Close Type     | SOTA | Tips |
| -------------- | ---- | ---- |
| CNN/Daily Mail |      |      |
| CBT            |      |      |
| CLOTH          |      |      |
| LAMBADA        |      |      |
| Who-did-What   |      |      |
| CliCR          |      |      |

| Mulit Choice | SOTA | Tips |
| ------------ | ---- | ---- |
| MCTest       |      |      |
| RACE         |      |      |
|              |      |      |

| Span Extraction                                          | SOTA                                                         | Tips |
| -------------------------------------------------------- | ------------------------------------------------------------ | ---- |
| SQuAD 1.1                                                |                                                              |      |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | BERT + DAE + AoA (ensemble) Joint Laboratory of HIT and iFLYTEK Research |      |
| NEWSQA                                                   |                                                              |      |
| TriviaQA                                                 |                                                              |      |
| DuoRC                                                    |                                                              |      |

| Free Answer                                     | SOTA                                                         | Tips |
| ----------------------------------------------- | ------------------------------------------------------------ | ---- |
| bAbI                                            |                                                              |      |
| [MARCO V2](http://www.msmarco.org/leaders.aspx) | Enriched BERT base + AOA index V1** Ming Yan of Alibaba Damo NLP |      |
| SearchQA                                        |                                                              |      |
| NarrativeQA                                     |                                                              |      |
| DuReader                                        |                                                              |      |

# Metric

- MAP
- MRR
- NDCG
- EM
- F1
- ROUGE-L
- BLEU

# General Architecture
+ Embedding
    + One-hot
    + Static Embedding
        + Word2Vec
        + Glove
    + Dynamic Embedding(Contextualized based)
        + Cove
        + ELMo
        + GPT
        + BERT
        + MASS
        + UniLM
        + XLNET
    + Multiple Granularity
        + Character Embedding
        + POS
        + NER
        + Binary Feature of Exact Match (EM)
        + Query-Category    
+ Feature Extraction
    + CNN
    + RNN
    + Transformer
+ Context-Question Interaction
    + Un Attn
    + Bi Attn
    + One-hop Interaction
    + Multi-hop Interaction
+ Answer Prediction
    + Word Predictor
    + Opinion Selector
    + Span Extractor
    + Answer Generator
    + **Ranker**

# Solutions

## Close Type

| Model | Tips | Result |
| ----- | ---- | ------ |
|       |      |        |
|       |      |        |
|       |      |        |

## Mulit Choice

| Model | Tips | Result |
| ----- | ---- | ------ |
|       |      |        |
|       |      |        |
|       |      |        |

## Span Extraction

| Model                                  | Tips | Result |
| -------------------------------------- | ---- | ------ |
| BIDAF                                  |      |        |
| Matching-LSTM                          |      |        |
| Hybird Attention-over-Attention Reader |      |        |

## Free Answer

| Model                                             | Tips                                                         | Result |
| ------------------------------------------------- | ------------------------------------------------------------ | ------ |
| Reading Wikipedia to Answer Open Domain Questions | http://www.zhuanzhi.ai/document/d6a0038bb0143a6805370adb58bf68be |        |
|                                                   |                                                              |        |
|                                                   |                                                              |        |

# Training Settings

# Problems

# Open Issues
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

- Reinforce Learning

- Answer Ranker

- Sentence Select

- KBMRC

# Milestone
+ pass

# Coding Standards

- 大小写
- 复数

# Usages

+ Will be used in “Workspace of Conversation-AI

# Reference

- Neural Machine Reading Comprehension: Methods and Trends 

- 2017年　以前的论文和数据集整理](https://www.zybuluo.com/ShawnNg/note/622592)
- [2018年　清华77篇机器阅读理解论文](http://www.zhuanzhi.ai/document/87418ceee95a21622d1d7a21f71a894a)
- 2019 年　有待整理
- [SQuAD 的一些模型]

