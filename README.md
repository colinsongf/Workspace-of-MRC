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
| MS-MARCO                                                 | ENG  | 100K  | 200K  | User Logs       | Web doc      | Summary by human       | MARCO 提供多篇来自搜索结果的网页文档，系统需要通过阅读这些文档来回答用户提出的问题。但是，文档中是否含有答案，以及答案具体在哪一篇文档中，都需要系统自己来判断解决。更有趣的是，有一部分问题无法在文档中直接找到答案，需要阅读理解模型自己做出判断；MARCO 也不限制答案必须是文档中的片段，很多问题的答案必须经过多篇文档综合提炼得到 |
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



======

# Question Answering

### Reference

+ 阅读理解与问答数据集 https://zhuanlan.zhihu.com/p/30308726
+ https://www.quora.com/Datasets-How-can-I-get-corpus-of-a-question-answering-website-like-Quora-or-Yahoo-Answers-or-Stack-Overflow-for-analyzing-answer-quality

### Papers with dataset

- **(NLVR)** A Corpus of Natural Language for Visual Reasoning, 2017 [[paper\]](http://yoavartzi.com/pub/slya-acl.2017.pdf) [[data\]](http://lic.nlp.cornell.edu/nlvr)
- **(MS MARCO)** MS MARCO: A Human Generated MAchine Reading COmprehension Dataset, 2016 [[paper\]](https://arxiv.org/abs/1611.09268) [[data\]](http://www.msmarco.org/)
- **(NewsQA)** NewsQA: A Machine Comprehension Dataset, 2016 [[paper\]](https://arxiv.org/abs/1611.09830) [[data\]](https://github.com/Maluuba/newsqa)
- **(SQuAD)** SQuAD: 100,000+ Questions for Machine Comprehension of Text, 2016 [[paper\]](http://arxiv.org/abs/1606.05250) [[data\]](http://stanford-qa.com/)
- **(GraphQuestions)** On Generating Characteristic-rich Question Sets for QA Evaluation, 2016 [[paper\]](http://cs.ucsb.edu/~ysu/papers/emnlp16_graphquestions.pdf) [[data\]](https://github.com/ysu1989/GraphQuestions)
- **(Story Cloze)** A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories, 2016 [[paper\]](http://arxiv.org/abs/1604.01696) [[data\]](http://cs.rochester.edu/nlp/rocstories)
- **(Children's Book Test)** The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations, 2015 [[paper\]](http://arxiv.org/abs/1511.02301) [[data\]](http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz)
- **(SimpleQuestions)** Large-scale Simple Question Answering with Memory Networks, 2015 [[paper\]](http://arxiv.org/pdf/1506.02075v1.pdf) [[data\]](https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz)
- **(WikiQA)** WikiQA: A Challenge Dataset for Open-Domain Question Answering, 2015 [[paper\]](http://research.microsoft.com/pubs/252176/YangYihMeek_EMNLP-15_WikiQA.pdf) [[data\]](http://research.microsoft.com/en-US/downloads/4495da01-db8c-4041-a7f6-7984a4f6a905/default.aspx)
- **(CNN-DailyMail)** Teaching Machines to Read and Comprehend, 2015 [[paper\]](http://arxiv.org/abs/1506.03340) [[code to generate\]](https://github.com/deepmind/rc-data) [[data\]](http://cs.nyu.edu/~kcho/DMQA/)
- **(QuizBowl)** A Neural Network for Factoid Question Answering over Paragraphs, 2014 [[paper\]](https://www.cs.umd.edu/~miyyer/pubs/2014_qb_rnn.pdf) [[data\]](https://www.cs.umd.edu/~miyyer/qblearn/index.html)
- **(MCTest)** MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text, 2013 [[paper\]](http://research.microsoft.com/en-us/um/redmond/projects/mctest/MCTest_EMNLP2013.pdf) [[data\]](http://research.microsoft.com/en-us/um/redmond/projects/mctest/data.html)[[alternate data link\]](https://github.com/mcobzarenco/mctest/tree/master/data/MCTest)
- **(QASent)** What is the Jeopardy model? A quasisynchronous grammar for QA, 2007 [[paper\]](http://homes.cs.washington.edu/~nasmith/papers/wang+smith+mitamura.emnlp07.pdf) [[data\]](

### what

- **http://cs.stanford.edu/people/mengqiu/data/qg-emnlp07-data.tgz)**

### MCTest

### Algebra

### Science

### Stanford Question Answering Dataset (SQuAD)

- official website
	- https://rajpurkar.github.io/SQuAD-explorer/
- reference
	- EMNLP2016 SQuAD:100,000+ Questions for Machine Comprehension of Text
		- https://arxiv.org/pdf/1606.05250.pdf
	- SQuAD，斯坦福在自然语言处理的野心
		- http://blog.csdn.net/jdbc/article/details/52514050
- SOTAs
	- Hybrid AoA Reader (ensemble)
		- Joint Laboratory of HIT and iFLYTEK Research
	- r-net + 融合模型
		- Microsoft Research Asia
	- SLQA + 融合模型
		- Alibaba iDST NLP
- detail
	- 这个竞赛基于SQuAD问答数据集，考察两个指标：EM和F1。
	- EM是指精确匹配，也就是模型给出的答案与标准答案一模一样；
	- F1，是根据模型给出的答案和标准答案之间的重合度计算出来的，也就是结合了召回率和精确率。
	- 目前阿里、微软团队并列第一，其中EM得分微软（r-net+融合模型）更高，F1得分阿里（SLQA+融合模型）更高。但是他们在EM成绩上都击败了“人类表现”
	- 一共有107,785问题，以及配套的 536 篇文章
		- 数据集的具体构建如下：
			    1. 文章是随机sample的wiki百科，一共有536篇wiki被选中。而每篇wiki，会被切成段落，最终生成了23215个自然段。之后就对这23215个自然段进行阅读理解，或者说自动问答。
			    2. 之后斯坦福，利用众包的方式，进行了给定文章，提问题并给答案的人工标注。他们将这两万多个段落给不同人，要求对每个段落提五个问题。
			    3. 让另一些人对提的这个问题用文中最短的片段给予答案，如果不会或者答案没有在文章中出现可以不给。之后经过他们的验证，人们所提的问题在问题类型分布上足够多样，并且有很多需要推理的问题，也就意味着这个集合十分有难度。如下图所示，作者列出了该数据集答案的类别分布，我们可以看到 日期，人名，地点，数字等都被囊括，且比例相当。
			    4. 这个数据集的评测标准有两个：
			        第一：F1
			        第二：EM。
			                EM是完全匹配的缩写，必须机器给出的和人给出的一样才算正确。哪怕有一个字母不一样，也会算错。而F1是将答案的短语切成词，和人的答案一起算recall，Precision和F1，即如果你match了一些词但不全对，仍然算分。
			    5. 为了这个数据集，他们还做了一个baseline，是通过提特征，用LR算法将特征组合，最终达到了40.4的em和51的f1。而现在IBM和新加坡管理大学利用深度学习模型，均突破了这个算法。可以想见，在不远的将来会有更多人对阅读理解发起挑战，自然语言的英雄也必将诞生。甚至会有算法超过人的准确度。

### MS MARCO

- 相比SQuAD，MARCO的挑战难度更大，因为它需要测试者提交的模型具备理解复杂文档、回答复杂问题的能力。
- 据了解，对于每一个问题，MARCO 提供多篇来自搜索结果的网页文档，系统需要通过阅读这些文档来回答用户提出的问题。但是，文档中是否含有答案，以及答案具体在哪一篇文档中，都需要系统自己来判断解决。更有趣的是，有一部分问题无法在文档中直接找到答案，需要阅读理解模型自己做出判断；MARCO 也不限制答案必须是文档中的片段，很多问题的答案必须经过多篇文档综合提炼得到。这对机器阅读理解提出了更高的要求，需要机器具备综合理解多文档信息、聚合生成问题答案的能力。

### NarrativeQA

- Deepmind 最新阅读理解数据集 NarrativeQA ，让机器挑战更复杂阅读理解问题
	- https://www.leiphone.com/news/201712/mjCYZ8WTiREqja6L.html
	- https://github.com/deepmind/narrativeqa
	- DeepMind认为目前的阅读理解数据集均存在着一定的局限性，包括：数据集小、不自然、只需要一句话定位回答的必须信息，等等。因而 Deepmind 认为，在这些数据集上的测试可能都是一个不能真实反映机器阅读理解能力的伪命题。

### The NarrativeQA Reading Comprehension Challenge

- 由 DeepMind 发布的全新机器阅读理解数据集 NarrativeQA，其难度和复杂度都进行了全面升级。
- 论文链接：https://www.paperweekly.site/papers/1397
- 代码链接：https://github.com/deepmind/narrativeqa

### SougoQA

- http://task.www.sogou.com/cips-sogou_qa/

### Graph Questions 

- On Generating Characteristic-rich Question Sets for QA Evaluation
- 文章发表在 EMNLP 2016，本文详细阐述了 GraphQuestions 这个数据集的构造方法，强调这个数据集是富含特性的（Characteristic-rich）。
	- 数据集特点：
		1. 基于 Freebase，有 5166 个问题，涉及 148 个不同领域；
		2. 从知识图谱中产生 Minimal Graph Queries，再将 Query 自动转换成规范化的问题；
		3. 由于 2，Logical Form 不需要人工标注，也不存在无法用 Logical Form 表示的问题；
		4. 使用人工标注的办法对问题进行 paraphrasing，使得每个问题有多种表述方式（答案不变），主要是 Entity-level Paraphrasing，也有 sentence-level；
		5. Characteristic-rich 指数据集提供了问题在下列维度的信息，使得研究者可以对问答系统进行细粒度的分析, 找到研究工作的前进方向：关系复杂度（Structure Complexity），普遍程度（Commonness），函数（Function），多重释义（Paraphrasing），答案候选数（Answer Cardinality）。
	- 论文链接：http://www.paperweekly.site/papers/906
	- 数据集链接：https://github.com/ysu1989/GraphQuestions

### LSDSem 2017 Shared Task: The Story Cloze Test

- Story Cloze Test：人工合成的完形填空数据集。
- 论文链接：http://www.paperweekly.site/papers/917
- 数据集链接：http://cs.rochester.edu/nlp/rocstories/

### Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering

- 百度深度学习实验室创建的中文开放域事实型问答数据集。
- 论文链接：http://www.paperweekly.site/papers/914
- 数据集链接：http://idl.baidu.com/WebQA.html

### Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems

- DeepMind 和牛津大学共同打造的代数问题数据集 AQuA（Algebra Question Answering）。
- 论文链接：http://www.paperweekly.site/papers/913
- 数据集链接：https://github.com/deepmind/AQuA

### Teaching Machines to Read and Comprehend

- DeepMind Q&A Dataset 是一个经典的机器阅读理解数据集，分为两个部分：
	1. CNN：~90k 美国有线电视新闻网（CNN）的新闻文章，~380k 问题；
	2. Daily Mail：~197k DailyMail 新闻网的新闻文章（不是邮件正文），~879k 问题。
- 论文链接：http://www.paperweekly.site/papers/915
- 数据集链接：http://cs.nyu.edu/~kcho/DMQA/

### Semantic Parsing on Freebase from Question-Answer Pairs

- 文章发表在 EMNLP-13，The Stanford NLP Group 是世界领先的 NLP 团队。
- 他们在这篇文章中引入了 WebQuestions 这个著名的问答数据集，WebQuestion 主要是借助 Google Suggestion 构造的
- 依靠 Freebase（一个大型知识图谱）中的实体来回答，属于事实型问答数据集（比起自然语言，容易评价结果优劣）
- 有 6642 个问答对
- 最初，他们构造这个数据集是为了做 Semantic Parsing，以及发布自己的系统 SEMPRE system。
- 论文链接：http://www.paperweekly.site/papers/827
- 数据集链接：http://t.cn/RWPdQQO

### A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories

- ROCStories dataset for story cloze test.
- 论文链接：http://www.paperweekly.site/papers/918
- 数据集链接：http://cs.rochester.edu/nlp/rocstories/

### MoleculeNet:  Benchmark for Molecular Machine Learning

- 一个分子机器学习 benchmark，最喜欢看到这种将机器学习应用到传统学科领域了。
- 论文链接：http://www.paperweekly.site/papers/862
- 数据集链接：http://t.cn/RWPda8r

### 关于维基百科文章的问答:Deepmind Question Answering Corpus

- https://github.com/deepmind/rc-data

### 关于亚马逊产品的问答

- Amazon question/answer data
	- http://jmcauley.ucsd.edu/data/amazon/qa/

### Looking Beyond the surface: A Chanllenge Set for Reading Comprehension over Multiple Sentences

- 特点:
	- 多选题
	- 问题的答案来自篇章中的多条语句
	- 数据集来自7个不同的领域
- 基准算法:
	- Random
	- IR
	- SurfaceIR
	- SemanticLP
	- BiDAF
- SOTA
	- SurfaceIR 结构的F1 值 相较人类结果 差 20个百分点

### 更多信息

- Datasets: How can I get corpus of a question-answering website like Quora or Yahoo Answers or Stack Overflow for analyzing answer quality?
	https://www.quora.com/Datasets-How-can-I-get-corpus-of-a-question-answering-website-like-Quora-or-Yahoo-Answers-or-Stack-Overflow-for-analyzing-answer-quality



文章发表在 EMNLP 2016，本文详细阐述了 GraphQuestions 这个数据集的构造方法，强调这个数据集是富含特性的（Characteristic-rich）。

- 数据集特点：
  1. 基于 Freebase，有 5166 个问题，涉及 148 个不同领域；
  2. 从知识图谱中产生 Minimal Graph Queries，再将 Query 自动转换成规范化的问题；
  3. 由于 2，Logical Form 不需要人工标注，也不存在无法用 Logical Form 表示的问题；
  4. 使用人工标注的办法对问题进行 paraphrasing，使得每个问题有多种表述方式（答案不变），主要是 Entity-level Paraphrasing，也有 sentence-level；
  5. Characteristic-rich 指数据集提供了问题在下列维度的信息，使得研究者可以对问答系统进行细粒度的分析, 找到研究工作的前进方向：关系复杂度（Structure Complexity），普遍程度（Commonness），函数（Function），多重释义（Paraphrasing），答案候选数（Answer Cardinality）。

