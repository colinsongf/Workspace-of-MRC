# Summary of Machine Reading Comprehension

# Target

+ Clean up current algorithms of Reading Comprehension and Question Answering
+ Read Paper and Write Note

# Reference

+ [2017年　以前的论文和数据集整理](https://www.zybuluo.com/ShawnNg/note/622592)
+ [2018年　清华77篇机器阅读理解论文](http://www.zhuanzhi.ai/document/87418ceee95a21622d1d7a21f71a894a)
+ 2019 年　有待整理
+ [SQuAD 的一些模型](<http://www.zhuanzhi.ai/document/d6a0038bb0143a6805370adb58bf68be>)
  + R-NET
  + SLQA

# LeaderBoard(sort by dataset)
+ SQuAD
    + BiDAF
    + r-NET
    + BERT
+ MARCO
+ LAMBADA
+ Who-did-What(WDW)
+ CNN & DailyMail(close style)
+ Children's Book Test
+ BookTest

# Metric

+ MAP
+ MRR
    + 是一个国际上通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，如果没有匹配的句子分数为0
    + 最终的分数为所有得分之和
+ NDCG
+ EM
+ F1
+ ROUGE-L
+ BLEU

# Idea 
+ Open domain
    + 检索
        + 检索得到主要的关键词
    + 答案生成
        + 模型将关键词输出为一句话
        

Doing

+ BERT/AOA

# Problems

+ pass

# Universial Framework

+ Answer Layer

# Papers(sort by type, year, dataset)

## Extractive QA

### [BiDAF-ICLR2017](https://arxiv.org/pdf/1611.01603.pdf)

![](https://ws1.sinaimg.cn/large/006tNc79ly1g1t3uklag5j31ak0t8gs0.jpg)
+ Abstract
  + 同时计算contex2query 和 query2context, 注意力的计算在时序上是独立的,并会flow到下一层
  + 避免了过早的summary 造成的信息丢失
+ Framework
  + char embedding:使用CNN训练得到,参考Kim(有待添加)

  + word embedding:Glove

  + 字向量和词向量拼接后,过一个Highway, 得到　Context X 和 query Y

  + Contextual Embedding Layer
    + 使用Bi-LSTM整合Highway的输出，表达词间关系
    + 输出
    	+ contex $ H \in R^{2d*T} $
    	+ query $ Q \in R^{d*J} $

  + Attention Flow Layer
    + 计算context和query词和词两两之间的相似性 
       $$ S_{tj} = \alpha(H_{:j}, U_{:j}) $$
       $$ \alpha = w^T_{S} [h;u;h \odot u] $$		

    + 计算context-to-query attention, 对于context中的词,按attention系数计算query中的词的 加权和 作为当前词的 **query aware representation**

      $$\alpha_t = softmax(St:) \in R^J​$$

      $$ \widetilde U_{:t} = \sum \alpha_{ij} U_{:j} R\in^{2d*J} ​$$

    + 计算query-to-context attention, 计算 query 和 每个 context 的最大相似度, query和context的相似度是query所有词里面和context相似度最大的, 然后计算context 的加权和

      $$ b = softmax(max_{col}(S)) ​$$
      $$ \widetilde{h} = \sum_t b_t H_{:t}  \in R^{2d}​$$
      $$ \widetilde{H} = tile(\widetilde{h})  ​$$	

    + final query-aware-representation of context

      $$ G_{:t} = \beta(H_{:t}, \widetilde U_{:t}, \widetilde H_{:t} ) $$

      $$ \beta(h;\widetilde{u};\widetilde{h}) = [h;\widetilde{u};h\odot\widetilde{u};h\odot\widetilde{h}] \in R^{8d}​$$	

  + Modeling Layer

    + 过Bi-LSTM 得到 M

  + Output Layer

    $$ p^1 = softmax(w^T_(p1)[G;M]​$$

    $$ p^2 = softmax(w^T_(p1)[G;M_2]$$

    $$ L(\theta) = - \frac{1}{N} \sum_i^{N} log(p^1_{y^1_i}) + log(p^2_{y^2_i})$$

  + results

    + SQuAD

    ![SQuAD](https://ws2.sinaimg.cn/large/006tNc79ly1g1w7p64a07j30k006e0tm.jpg)

    + CNN/DailyMail

    ![CNN/Dialy Mail](https://ws3.sinaimg.cn/large/006tNc79ly1g1w7q415szj30k00asdh8.jpg)

### [Matching-LSTM](https://arxiv.org/pdf/1608.07905.pdf)

![M-LSTM](https://ws2.sinaimg.cn/large/006tNc79ly1g1t3urvg0fj318k0p4ajq.jpg)
+ Abstract
  + LSTM 编码原文的上下文信息
  + Match-LSTM 匹配原文和问题
  + Answer-Pointer :　使用Ptr网络, 预测答案
  	+ Sequence Model :　答案是不连续的
  	+ Boundary Model : 答案是连续的, 在SQuAD数据集上 Boundary 比 Sequence 的效果要好

+ Framework
  + LSTM Preprocessing Layer

    + 使用embedding表示Question 和 Context, 在使用单向LSTM编码,得到hidden state 表示 , l 是隐变量长度

      $$ H^P = \vec{LSTM}(P) \in R^{l *P}​$$

      $$H^Q = \vec {LSTM}(Q) \in R^{l *Q}​$$

  + Match-LSTM Layer
    + 类似文本蕴含:前提H, 假设T, M-LSTM序列化的经过假设的每一个词,然后预测前提是否继承自假设
    + 文本问答中, question 当做 H, context当做T, 可以看成带着问题去段落中找答案(利用soft-attention)

  + Answer Pointer Layer

  	+ Sequence Model
  	+ Boundary Model

+ Results
  ![](https://img.mukewang.com/5ac37472000179ad15620702.png)  	  


### [R-NET ACL 2017 MSRA](<https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf>)

![](https://ws3.sinaimg.cn/large/006tKfTcly1g1pcrgzkffj318p0tctc0.jpg)

+ Abstract
  + 借鉴 Match-LSTM 和 Pointer Network的思想
+ Framework
  + Question and Passage Encoder
    + Word Embedding(Glove) + Char Embedding 
    + Question Encoder $$u^Q_t$$
    + Passage Encoder $$u_t^Q$$
  + Gated Attention-Based Recurrent Network
  + Self-Matching Attention
  + Output Layer

## Search QA

## Generative QA