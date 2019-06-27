# View of Question Answering

### ACL 2018 | 北大与百度提出多文章验证方法：让机器验证阅读理解候选答案
+ https://zhuanlan.zhihu.com/p/36925804

#### 基于知识的智能问答技术　冯岩松
#### 基于深度学习的阅读理解　冯岩松

### 近期有哪些值得读的QA论文？| 专题论文解读(作者丨徐阿衡 学校丨卡耐基梅隆大学硕士)
+ https://www.jiqizhixin.com/articles/2018-06-11-14

### (FastQA) Making Neural QA as Simple as Possible but not Simpler
+ 论文 | Making Neural QA as Simple as Possible but not Simpler
+ 链接 | https://www.paperweekly.site/papers/835
+ 作者 | Dirk Weissenborn / Georg Wiese / Laura Seiffe
+ 阅读理解系列的框架很多大同小异，但这篇 paper 真心觉得精彩，虽然并不是最新最 state-of-art。

### GDAN
- 论文 | Semi-Supervised QA with Generative Domain-Adaptive Nets
- 链接 | https://www.paperweekly.site/papers/576
- 作者 | Zhilin Yang / Junjie Hu / Ruslan Salakhutdinov / William W. Cohen
- GDAN，Question Generation 和 Question Answering 相结合，利用少量的有标注的 QA 对 + 大量的无标注的 QA 对来训练 QA 模型。

### QANet
- 论文 | QANet - Combining Local Convolution with Global Self-Attention for Reading Comprehension
- 链接 | https://www.paperweekly.site/papers/1901
- 源码 | https://github.com/NLPLearn/QANet
- **CMU 和 Google Brain 新出的文章，SQuAD 目前的并列第一，两大特点： **
	- 1. 模型方面创新的用 CNN+attention 来完成阅读理解任务。
		- 在编码层放弃了 RNN，只采用 CNN 和 self-attention。CNN 捕捉文本的局部结构信息（ local interactions），self-attention 捕捉全局关系（ global interactions），在没有牺牲准确率的情况下，加速了训练（训练速度提升了 3x-13x，预测速度提升 4x-9x）。


	- 2. 数据增强方面通过神经翻译模型（把英语翻译成外语（德语/法语）再翻译回英语）的方式来扩充训练语料，增加文本多样性。
		- 其实目前多数 NLP 的任务都可以用 word vector + RNN + attention 的结构来取得不错的效果，虽然我挺偏好 CNN 并坚定相信 CNN 在 NLP 中的作用（捕捉局部相关性&方便并行），但多数情况下也是跟着主流走并没有完全舍弃过 RNN，这篇论文还是给了我们很多想象空间的。





### SQuAD 数据集的各种解决方案
+ https://www.sohu.com/a/142040203_500659

### AAAI 2018论文解读 | 基于文档级问答任务的新注意力模型
- 论文 | A Question-Focused Multi-Factor Attention Network for Question Answering
- 链接 | https://www.paperweekly.site/papers/1597
- 源码 | https://github.com/nusnlp/amanda

# 检索问答
### 检索式问答系统的语义匹配模型（神经网络篇）
+ https://zhuanlan.zhihu.com/p/26879507

### PaperWeekly 第37期 | 论文盘点：检索式问答系统的语义匹配模型（神经网络篇）
- https://zhuanlan.zhihu.com/p/26879507

### Multi-Cast Attention Networks for Retrieval-based Question Answering and Response Prediction
+ 南洋理工 KDD2018
+ 论文在问答和对话建模方向为各种检索和匹配任务提出了一个通用神经网络排序模型
+ 引入attention 作为pooling操作, 并将其作为一种特征增强方法

### Learning to Ask Good Question:Ranking Clarification Questions using Neural Expected Value of Perfect Information
+ 马里兰 ACL2018
+ 论文基于完全信息期望值(EVPI,expected value with perfect information)架构构建了一个用于解决澄清问题排序的NN模型
+ 并利用问答网站"StackExchange"构建了一个三元组(post,question,answer)数据集,用于训练一个能够根据提问者所提出的问题来给出澄清问题的模型

### QA相关资源/数据集/论文列表 (2017-5-25)
+ https://blog.csdn.net/amds123/article/details/72758936

### IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models
+ 在现代信息检索领域一直是两大学派之争的局面。一方面，经典思维流派是假设在文档和信息需求（由查询可知）之间存在着一个独立的随机生成过程。另一方面，现代思维流派则充分利用机器学习的优势，将文档和搜索词联合考虑为特征，并从大量训练数据中预测其相关性或排序顺序标签。
+ 本篇 SIGIR2017 的满分论文则首次提出将两方面流派的数据模型通过一种对抗训练的方式统一在一起，使得两方面的模型能够相互提高，最终使得检索到的文档更加精准。文章的实验分别在网络搜索、推荐系统以及问答系统三个应用场景中实现并验证了结果的有效性。

### 机器这次击败人之后，争论一直没平息 | SQuAD风云
+ https://zhuanlan.zhihu.com/p/33124445

### 12 papers to understand QA system with Deep Learning
- http://blog.csdn.net/abcjennifer/article/details/51232645

### A Question-Focused Multi-Factor Attention Network for Question Answering
- https://www.paperweekly.site/papers/1597
- https://github.com/nusnlp/amanda

### Fast and Accurate Reading Comprehension by Combining Self-Attention and Convolution
-《Fast and Accurate Reading Comprehension by Combining Self-attention and Convolution》阅读笔记
- 本文是 CMU 和 Google Brain 发表于 ICLR 2018 的文章，论文改变了以往机器阅读理解均使用 RNN 进行建模的习惯，使用卷积神经网络结合自注意力机制，完成机器阅读理解任务。
- 其中作者假设，卷积神经网络可建模局部结构信息，而自注意力机制可建模全文互动（Interaction）关系，这两点就足以完成机器阅读理解任务。
- 论文链接
	- https://www.paperweekly.site/papers/1759


### 百度NLP团队登顶MARCO阅读理解测试排行榜
- http://tech.qq.com/a/20180222/008569.htm
- 使用了一种新的多候选文档联合建模表示方法，通过注意力机制使不同文档产生的答案之间能够产生交换信息，互相印证，从而更好的预测答案。据介绍，此次百度只凭借单模型（single model）就拿到了第一名，并没有提交更容易拿高分的多模型集成（ensemble）结果

# 社区问答
### Attentive Recurrent Tensor Model for Community Question Answering
- 社区问答有一个很主要的挑战就是句子间词汇与语义的鸿沟。本文使用了 phrase-level 和 token-level 两个层次的 attention 来对句子中的词赋予不同的权重，并参照 CNTN 模型用神经张量网络计算句子相似度的基础上，引入额外特征形成 3-way 交互张量相似度计算。
	- 围绕答案选择、最佳答案选择、答案触发三个任务，论文提出的模型 RTM 取得了多个 state-of-art 效果。
	- 论文链接 : https://www.paperweekly.site/papers/1741
 
# Knowledge base
### 「知识表示学习」专题论文推荐 | 每周论文清单
- https://zhuanlan.zhihu.com/p/33606964

### 知识图谱与知识表征学习系列
- https://zhuanlan.zhihu.com/p/27664263

### 怎么利用知识图谱构建智能问答系统？
+ https://www.zhihu.com/question/30789770/answer/116138035
+ https://zhuanlan.zhihu.com/p/25735572

### 揭开知识库问答KB-QA的面纱1·简介篇
+ 什么是知识库（knowledge base, KB）
+ 什么是知识库问答（knowledge base question answering, KB-QA）
+ 知识库问答的主流方法
+ 知识库问答的数据集

### 经典论文解读 | 基于Freebase的问答研究
- 本文给出了一种 end-to-end 的系统来自动将 NL 问题转换成 SPARQL 查询语言。
- 作者综合了**实体识别**以及**距离监督**和 **learning-to-rank** 技术，使得 QA 系统的精度提高了不少，整个过程介绍比较详细，模型可靠接地气。
- 本文要完成的任务是根据 **KB** 知识来回答自然语言问题，给出了一个叫 Aqqu 的系统，首先为问题生成一些备选 query，然后使用学习到的模型来对这些备选 query 进行排名，返回排名最高的 query
- 论文 | More Accurate Question Answering on Freebase
- 链接 | https://www.paperweekly.site/papers/1356
- 源码 | https://github.com/ad-freiburg/aqqu

### Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks
- 传统 QA 问题的解决方法是从知识库或者生文本中推测答案，本文将通用模式扩展到自然语言 QA 的应用当中，采用记忆网络来关注文本和 KB 相结合的大量事实。
- 论文链接
	- https://www.paperweekly.site/papers/1734
- 代码链接
	- https://github.com/rajarshd/TextKBQA


## Some Projects
### LSTM 中文
- https://github.com/S-H-Y-GitHub/QA
- 本项目通过建立双向长短期记忆网络模型，实现了在多个句子中找到给定问题的答案所在的句子这一功能。在使用了互联网第三方资源的前提下，用training.data中的数据训练得到的模型对develop.data进行验证，MRR可达0.75以上
