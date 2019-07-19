[TOC]

# Workspace of Machine Reading Comprehension

# Target

+ Algorithms implementation of **M**achine **R**eading **C**omprehension 
+ Efficient and beautiful code
+ General Architecture for MRC
    + Framework for multi-dataset and multi-card
    + Framework for model ensemble

# Dataset

| Dataset                                                  | Lang | Query | Doc   | query source    | doc source   | Answer                 | SOTA                                                         |
| -------------------------------------------------------- | ---- | ----- | ----- | --------------- | ------------ | ---------------------- | ------------------------------------------------------------ |
| CNN/Daily Mail                                           | ENG  | 1.4M  | 300K  | Close           | News         | Fill in entity         |                                                              |
| CBT                                                      |      |       |       |                 |              |                        |                                                              |
| CLOTH                                                    |      |       |       |                 |              |                        |                                                              |
| LAMBADA                                                  |      |       |       |                 |              |                        |                                                              |
| Who-did-What                                             |      |       |       |                 |              |                        |                                                              |
| CliCR                                                    |      |       |       |                 |              |                        |                                                              |
| HLF-RC                                                   | CHN  | 100K  | 28K   | Close           | Fairy/news   | Fill in word           |                                                              |
| RACE                                                     | ENG  | 870K  | 50K   | english exam    | english exam | Multi choice           |                                                              |
| MCTest                                                   |      |       |       |                 |              | Multi choice           |                                                              |
| SQuAD 1                                                  | ENG  | 100K  | 536   |                 |              | Span                   | BERT + DAE + AoA (ensemble) Joint Laboratory of HIT and iFLYTEK Research |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) |      |       |       |                 |              | Span                   |                                                              |
| NEWSQA                                                   |      |       |       |                 |              | span                   |                                                              |
| TriviaQA                                                 | ENG  | 40K   | 660K  | Trivia websites | Wiki/Web doc | Span/substring of word |                                                              |
| DuoRC                                                    |      |       |       |                 |              | Span                   |                                                              |
| MS-MARCO                                                 | ENG  | 100K  | 200K  | User Logs       | Web doc      | Summary by human       |                                                              |
| DuReader                                                 | ENG  | 200K  | 1000K | User Logs       | Web doc/CQA  | Summary  by human      |                                                              |
|                                                          |      |       |       |                 |              |                        |                                                              |

| Free Answer                                     | SOTA                                                         | Tips |
| ----------------------------------------------- | ------------------------------------------------------------ | ---- |
| bAbI                                            |                                                              |      |
| [MARCO V2](http://www.msmarco.org/leaders.aspx) | Enriched BERT base + AOA index V1** Ming Yan of Alibaba Damo NLP |      |
| SearchQA                                        |                                                              |      |
| NarrativeQA                                     |                                                              |      |
| DuReader                                        |                                                              |      |

| Factoid QA                                                   | SOTA | Tips                                                         |
| ------------------------------------------------------------ | ---- | ------------------------------------------------------------ |
| https://github.com/brmson/dataset-factoid-webquestions http://www.cs.cmu.edu/~ark/QA-data |      |                                                              |
| [https://www.microsoft.com/en-us/research/publication/the-use-of-external-knowledge-of-factoid-qa/](https://www.microsoft.com/en-us/research/publication/the-use-of-external-knowledge-of-factoid-qa/) |      | Generating factoid questions with recurrent neural networks: The 30m factoid question-answer corpus  将 KB 三元组转化为问句 |
|                                                              |      |                                                              |

| Retrieval QA | SOTA | Tips |
| ------------ | ---- | ---- |
|              |      |      |
|              |      |      |
|              |      |      |

| KBQA                                                 | SOTA | Tips |
| ---------------------------------------------------- | ---- | ---- |
| 知识库 :　Freebase, DBpedia, Wikipedia, yago, satori |      |      |
| ATIS                                                 |      |      |
| GeoQuery                                             |      |      |
| QALD Series                                          |      |      |
| Free 917                                             |      |      |
| Web Questions                                        |      |      |
| Web QuestionsSP                                      |      |      |
| SimpleQuestions                                      |      |      |
| WikiMovies                                           |      |      |
| TriviaQA                                             |      |      |



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

| Close Type | Tips |
| ---------- | ---- |
|            |      |
|            |      |
|            |      |

| Multi Choice | Tips |
| ------------ | ---- |
|              |      |
|              |      |
|              |      |

| Span Extraction                        | Tips |
| -------------------------------------- | ---- |
| BIDAF                                  |      |
| Matching-LSTM                          |      |
| Hybird Attention-over-Attention Reader |      |
| DrQA                                   |      |
| QANet                                  |      |

| Free Answer                                       | Tips                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ |
| Reading Wikipedia to Answer Open Domain Questions | http://www.zhuanzhi.ai/document/d6a0038bb0143a6805370adb58bf68be |
|                                                   |                                                              |
|                                                   |                                                              |

| Factoid QA | Tips |
| ---------- | ---- |
|            |      |
|            |      |
|            |      |

| Retrieval QA | Tips |
| ------------ | ---- |
|              |      |
|              |      |
|              |      |

| KBQA | Tips |
| ---- | ---- |
|      |      |
|      |      |
|      |      |



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
- DuReader ：https://zhuanlan.zhihu.com/p/36415104

