### [BiDAF-ICLR2017](https://arxiv.org/pdf/1611.01603.pdf)

![](https://ws1.sinaimg.cn/large/006tNc79ly1g1t3uklag5j31ak0t8gs0.jpg)

- Abstract

  - 同时计算contex2query 和 query2context, 注意力的计算在时序上是独立的,并会flow到下一层
  - 避免了过早的summary 造成的信息丢失

- Framework

  - char embedding:使用CNN训练得到,参考Kim(有待添加)

  - word embedding:Glove

  - 字向量和词向量拼接后,过一个Highway, 得到　Context X 和 query Y

  - Contextual Embedding Layer

    - 使用Bi-LSTM整合Highway的输出，表达词间关系
    - 输出
      - contex $ H \in R^{2d*T} $
      - query $ Q \in R^{d*J} $

  - Attention Flow Layer

    - 计算context和query词和词两两之间的相似性 
      $$ S_{tj} = \alpha(H_{:j}, U_{:j}) $$
      $$ \alpha = w^T_{S} [h;u;h \odot u] $$		

    - 计算context-to-query attention, 对于context中的词,按attention系数计算query中的词的 加权和 作为当前词的 **query aware representation**

      $$\alpha_t = softmax(St:) \in R^J$$

      $$ \widetilde U_{:t} = \sum \alpha_{ij} U_{:j} R\in^{2d*J} $$

    - 计算query-to-context attention, 计算 query 和 每个 context 的最大相似度, query和context的相似度是query所有词里面和context相似度最大的, 然后计算context 的加权和

      $$ b = softmax(max_{col}(S)) $$
      $$ \widetilde{h} = \sum_t b_t H_{:t}  \in R^{2d}$$
      $$ \widetilde{H} = tile(\widetilde{h})  $$	

    - final query-aware-representation of context

      $$ G_{:t} = \beta(H_{:t}, \widetilde U_{:t}, \widetilde H_{:t} ) $$

      $$ \beta(h;\widetilde{u};\widetilde{h}) = [h;\widetilde{u};h\odot\widetilde{u};h\odot\widetilde{h}] \in R^{8d}$$	

  - Modeling Layer

    - 过Bi-LSTM 得到 M

  - Output Layer

    $$ p^1 = softmax(w^T_(p1)[G;M]$$

    $$ p^2 = softmax(w^T_(p1)[G;M_2]$$

    $$ L(\theta) = - \frac{1}{N} \sum_i^{N} log(p^1_{y^1_i}) + log(p^2_{y^2_i})$$

  - results

    - SQuAD

    ![SQuAD](https://ws2.sinaimg.cn/large/006tNc79ly1g1w7p64a07j30k006e0tm.jpg)

    - CNN/DailyMail

    ![CNN/Dialy Mail](https://ws3.sinaimg.cn/large/006tNc79ly1g1w7q415szj30k00asdh8.jpg)

