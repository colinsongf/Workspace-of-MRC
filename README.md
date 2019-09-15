[TOC]

# Workspace of Machine Reading Comprehension

# Target

+ Algorithms implementation of **M**achine **R**eading **C**omprehension 
+ Efficient and beautiful code
+ General Architecture for MRC
    + Framework for multi-dataset and multi-card
    + Framework for model ensemble

# Dataset

## Challenge

| New        | Tips  |
| ---------- | ----- |
| CIPS-SOGOU | DGCNN |
|            |       |
|            |       |



## Key Dataset

| Mulit Type                                                   | Types |      |
| ------------------------------------------------------------ | ----- | ---- |
| MT1 百度深度学习实验室;中文开放域事实型问答数据集 [paper](Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering) [dataset]() |       |      |
|                                                              |       |      |
|                                                              |       |      |



## Close/Factoid/Multi Choice

| Dataset                                                      | Lang | Query             | Doc              | query source                    | doc source   | Type                                                         |
| ------------------------------------------------------------ | ---- | ----------------- | ---------------- | ------------------------------- | ------------ | ------------------------------------------------------------ |
| Deep Mind [CNN/Daily Mail](http://cs.nyu.edu/~kcho/DMQA/)    | ENG  | 1.4M(380K + 879K) | 300K(90k + 197K) | Close                           | News         | Fill in entity                                               |
| CBT                                                          |      |                   |                  |                                 |              |                                                              |
| CLOTH                                                        |      |                   |                  |                                 |              |                                                              |
| LAMBADA                                                      |      |                   |                  |                                 |              |                                                              |
| Who-did-What                                                 |      |                   |                  |                                 |              |                                                              |
| CliCR                                                        |      |                   |                  |                                 |              |                                                              |
| HLF-RC                                                       | CHN  | 100K              | 28K              | Close                           | Fairy/news   | Fill in word                                                 |
| RACE                                                         | ENG  | 870K              | 50K              | english exam                    | english exam | Multi choice                                                 |
| MCTest                                                       |      |                   |                  |                                 |              | Multi choice                                                 |
| SQuAD 1                                                      | ENG  | 100K              | 536              |                                 |              | Span                                                         |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)     |      |                   |                  |                                 |              | Span                                                         |
| NEWSQA                                                       |      |                   |                  |                                 |              | span                                                         |
| TriviaQA                                                     | ENG  | 40K               | 660K             | Trivia websites                 | Wiki/Web doc | Span/substring of word                                       |
| DuoRC                                                        |      |                   |                  |                                 |              | Span                                                         |
| MS-MARCO<br />答案不限制在文中；多文档                       | ENG  | 100K              | 200K             | User Logs                       | Web doc      | Summary by human                                             |
| DuReader                                                     | ENG  | 200K              | 1000K            | User Logs                       | Web doc/CQA  | Summary  by human                                            |
| MRQA2019                                                     |      |                   |                  |                                 |              |                                                              |
| 哈工大-讯飞<br />CMRC 2018<br />形式与SQuAD相同              |      |                   |                  |                                 |              |                                                              |
| 中国台湾台达研究院<br />DRCD<br />形式与SQuAD相同，是基于繁体中文的抽取式阅读理解数据集 |      |                   |                  |                                 |              |                                                              |
| 亚马逊产品的问答 [data](http://jmcauley.ucsd.edu/data/amazon/qa/) |      |                   |                  |                                 |              |                                                              |
| [Deepmind Question Answering Corpus](https://github.com/deepmind/rc-data) |      |                   |                  |                                 |              |                                                              |
| DeepMind 和牛津大学<br />Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems [dataset](https://github.com/deepmind/AQuA) |      |                   |                  |                                 |              |                                                              |
| LSDSem 2017 Shared Task: The Story Cloze Test [dataset](http://cs.rochester.edu/nlp/rocstories/) |      |                   |                  | Close                           |              |                                                              |
| [SougoQA](http://task.www.sogou.com/cips-sogou_qa/)          |      |                   |                  |                                 |              |                                                              |
| CMU [factoid webquestion](https://github.com/brmson/dataset-factoid-webquestions)<br/><br/>[QA-Data](http://www.cs.cmu.edu/~ark/QA-data) |      |                   |                  | Factoid QA                      |              |                                                              |
| microsoft<br />[use-of-external-knowledge-of-factoid-qa/](https://www.microsoft.com/en-us/research/publication/the-use-of-external-knowledge-of-factoid-qa/)<br />将 KB 三元组转化为问句 |      |                   |                  | Factoid<br />External Knowledge |              |                                                              |
| Looking Beyond the surface                                   |      |                   |                  | Mulit Choice QA                 |              | 多选题 问题的答案来自篇章中的多条语句 数据集来自7个不同的领域 |

## Free Answer

| Free Answer                                     | SOTA                                                         | Tips                                                         |
| ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| bAbI                                            |                                                              |                                                              |
| [MARCO V2](http://www.msmarco.org/leaders.aspx) | Enriched BERT base + AOA index V1** Ming Yan of Alibaba Damo NLP |                                                              |
| SearchQA                                        |                                                              |                                                              |
| NarrativeQA                                     | https://www.leiphone.com/news/201712/mjCYZ8WTiREqja6L.html https://github.com/deepmind/narrativeqa | DeepMind认为目前的阅读理解数据集均存在着一定的局限性，包括：数据集小、不自然、只需要一句话定位回答的必须信息，等等。因而 Deepmind 认为，在这些数据集上的测试可能都是一个不能真实反映机器阅读理解能力的伪命题。 |
| DuReader                                        |                                                              |                                                              |
|                                                 |                                                              |                                                              |

## KBQA

| KBQA                                                         | Tips                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 知识库 :　Freebase, DBpedia, Wikipedia, yago, satori         |                                                              |
| ATIS                                                         |                                                              |
| GeoQuery                                                     |                                                              |
| QALD Series                                                  |                                                              |
| Free 917                                                     |                                                              |
| Web Questions                                                |                                                              |
| Web QuestionsSP                                              |                                                              |
| SimpleQuestions                                              |                                                              |
| WikiMovies                                                   |                                                              |
| TriviaQA                                                     |                                                              |
| [Web Question]()Semantic Parsing on Freebase from Question-Answer Pairs<br />[dataset](http://t.cn/RWPdQQO) | Standford;EMNLP-13 他们在这篇文章中引入了 WebQuestions 这个著名的问答数据集，WebQuestion 主要是借助 Google Suggestion 构造的 依靠 Freebase（一个大型知识图谱）中的实体来回答，属于事实型问答数据集（比起自然语言，容易评价结果优劣） 有 6642 个问答对 最初，他们构造这个数据集是为了做 Semantic Parsing，以及发布自己的系统 SEMPRE system |
| Graph Questions <br />[paper](On Generating Characteristic-rich Question Sets for QA Evaluation) [dataset](https://github.com/ysu1989/GraphQuestions) | 文章发表在 EMNLP 2016，本文详细阐述了 GraphQuestions 这个数据集的构造方法，强调这个数据集是富含特性的（Characteristic-rich）。  数据集特点：  基于 Freebase，有 5166 个问题，涉及 148 个不同领域； 从知识图谱中产生 Minimal Graph Queries，再将 Query 自动转换成规范化的问题； 由于 2，Logical Form 不需要人工标注，也不存在无法用 Logical Form 表示的问题； 使用人工标注的办法对问题进行 paraphrasing，使得每个问题有多种表述方式（答案不变），主要是 Entity-level Paraphrasing，也有 sentence-level； Characteristic-rich 指数据集提供了问题在下列维度的信息，使得研究者可以对问答系统进行细粒度的分析, 找到研究工作的前进方向：关系复杂度（Structure Complexity），普遍程度（Commonness），函数（Function），多重释义（Paraphrasing），答案候选数（Answer Cardinality） |
| NLPCC-DBQA                                                   | NLPCC 2016 的评测任务;                                       |

| RRC                                  | SOTA | Tips                                                         |
| ------------------------------------ | ---- | ------------------------------------------------------------ |
| https://arxiv.org/pdf/1904.02232.pdf |      | this paper explores the potential of turning customer reviews into a large source of knowledge that can be exploited to answer user questions |
|                                      |      |                                                              |
|                                      |      |                                                              |



# Metric

- MAP
- MRR
- NDCG
- EM
- F1
- ROUGE-L
- BLEU

# General Architecture

## Embedding

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

## Feature Extraction

+ CNN
+ RNN
+ Transformer

## Context-Question Interaction

+ Un Attn
+ Bi Attn
+ One-hop Interaction
+ Multi-hop Interaction

## Answer Prediction

+ Word Predictor
+ Opinion Selector
+ Span Extractor
+ Answer Generator
+ **Ranker**

# Solutions

## Close Type

| Close Type                                                   | Tips                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories | ROCStories dataset for story cloze test. 论文链接：http://www.paperweekly.site/papers/918 数据集链接：http://cs.rochester.edu/nlp/rocstories/ |
|                                                              |                                                              |

## Multi Choice

| Multi Choice                                                 | Tips                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Looking Beyond the surface: A Chanllenge Set for Reading Comprehension over Multiple Sentences | 特点:  多选题 问题的答案来自篇章中的多条语句 数据集来自7个不同的领域 基准算法:  Random IR SurfaceIR SemanticLP BiDAF SOTA  SurfaceIR 结构的F1 值 相较人类结果 差 20个百分点 |
|                                                              |                                                              |

## Span Extraction

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

## Factoid QA

| Factoid QA | Tips |
| ---------- | ---- |
|            |      |
|            |      |
|            |      |

## Retrieval QA

| Retrieval QA | Tips |
| ------------ | ---- |
|              |      |
|              |      |
|              |      |

## KBQA

| KBQA | Tips |
| ---- | ---- |
|      |      |
|      |      |
|      |      |

## NLU + NLG

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |



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

# Usages

+ Will be used in “Workspace of Conversation-AI

# Reference

- Neural Machine Reading Comprehension: Methods and Trends 
- 2017年　以前的论文和数据集整理](https://www.zybuluo.com/ShawnNg/note/622592)
- [2018年　清华77篇机器阅读理解论文](http://www.zhuanzhi.ai/document/87418ceee95a21622d1d7a21f71a894a)
- 2019 年　有待整理
- DuReader ：https://zhuanlan.zhihu.com/p/36415104

## task

- NLPCC2017的**Task5：Open Domain Question Answering**
  - 基于该数据集实现的论文  http://www.doc88.com/p-9095635489643.html
  - NLPCC比赛数据集下载页面
    - http://tcci.ccf.org.cn/conference/2017/taskdata.php 
    - http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html 

## project

### BERT-KBQA

- https://github.com/jkszw2014/bert-kbqa-NLPCC2017
- https://github.com/jkszw2014/bert-kbqa-NLPCC2017/tree/master/NER_BERT-BiLSTM-CRF

## dataset

- 阅读理解与问答数据集 https://zhuanlan.zhihu.com/p/30308726
- Datasets: How can I get corpus of a question-answering website like Quora or Yahoo Answers or Stack Overflow for analyzing answer quality?
  https://www.quora.com/Datasets-How-can-I-get-corpus-of-a-question-answering-website-like-Quora-or-Yahoo-Answers-or-Stack-Overflow-for-analyzing-answer-quality

## paper

+ Adversarial Domain Adaptation for Machine Reading Comprehension
  + 对抗领域自适应阅读理解
  + EMNLP 2019
+ Discourse-Aware Semantic Self-Attention for Narritive Reading Comprehension
  + EMNLP-IJCNLP 2019
  + 语义自注意叙事阅读理解
+ Incorporating Relation Knowledge into Commonsense Reading Comprehension with Multi-task Learning
  + CIKM 2019 阿里
  + 嵌入关系知识到常规知识图谱
+ **Machine Reading Comprehension: a Literature Review**
  + MRC 综述
  + https://arxiv.org/abs/1907.01686v1
+ Natural Reading Comprehension: Methods and Trends
  + https://arxiv.org/abs/1907.01118
+ **(NLVR)** A Corpus of Natural Language for Visual Reasoning, 2017 [[paper\]](http://yoavartzi.com/pub/slya-acl.2017.pdf) [[data\]](http://lic.nlp.cornell.edu/nlvr)
+ **(MS MARCO)** MS MARCO: A Human Generated MAchine Reading COmprehension Dataset, 2016 [[paper\]](https://arxiv.org/abs/1611.09268) [[data\]](http://www.msmarco.org/)
+ **(NewsQA)** NewsQA: A Machine Comprehension Dataset, 2016 [[paper\]](https://arxiv.org/abs/1611.09830) [[data\]](https://github.com/Maluuba/newsqa)
+ **(SQuAD)** SQuAD: 100,000+ Questions for Machine Comprehension of Text, 2016 [[paper\]](http://arxiv.org/abs/1606.05250) [[data\]](http://stanford-qa.com/)
+ **(GraphQuestions)** On Generating Characteristic-rich Question Sets for QA Evaluation, 2016 [[paper\]](http://cs.ucsb.edu/~ysu/papers/emnlp16_graphquestions.pdf) [[data\]](https://github.com/ysu1989/GraphQuestions)
+ **(Story Cloze)** A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories, 2016 [[paper\]](http://arxiv.org/abs/1604.01696) [[data\]](http://cs.rochester.edu/nlp/rocstories)
+ **(Children's Book Test)** The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations, 2015 [[paper\]](http://arxiv.org/abs/1511.02301) [[data\]](http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz)
+ **(SimpleQuestions)** Large-scale Simple Question Answering with Memory Networks, 2015 [[paper\]](http://arxiv.org/pdf/1506.02075v1.pdf) [[data\]](https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz)
+ **(WikiQA)** WikiQA: A Challenge Dataset for Open-Domain Question Answering, 2015 [[paper\]](http://research.microsoft.com/pubs/252176/YangYihMeek_EMNLP-15_WikiQA.pdf) [[data\]](http://research.microsoft.com/en-US/downloads/4495da01-db8c-4041-a7f6-7984a4f6a905/default.aspx)
+ **(CNN-DailyMail)** Teaching Machines to Read and Comprehend, 2015 [[paper\]](http://arxiv.org/abs/1506.03340) [[code to generate\]](https://github.com/deepmind/rc-data) [[data\]](http://cs.nyu.edu/~kcho/DMQA/)
+ **(QuizBowl)** A Neural Network for Factoid Question Answering over Paragraphs, 2014 [[paper\]](https://www.cs.umd.edu/~miyyer/pubs/2014_qb_rnn.pdf) [[data\]](https://www.cs.umd.edu/~miyyer/qblearn/index.html)
+ **(MCTest)** MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text, 2013 [[paper\]](http://research.microsoft.com/en-us/um/redmond/projects/mctest/MCTest_EMNLP2013.pdf) [[data\]](http://research.microsoft.com/en-us/um/redmond/projects/mctest/data.html)[[alternate data link\]](https://github.com/mcobzarenco/mctest/tree/master/data/MCTest)
+ **(QASent)** What is the Jeopardy model? A quasisynchronous grammar for QA, 2007 [[paper\]]



