# Search QA

### Reference
- 检索式问答系统的语义匹配模型
	- https://zhuanlan.zhihu.com/p/26879507

### 检索式问答系统的语义匹配模型
+ https://zhuanlan.zhihu.com/p/26879507
+ 实现方式
	+ 问答系统可以基于规则实现
	+ 可以基于检索实现
	+ 还可以通过对 query 进行解析或语义编码来生成候选回复
	+ 如通过解析 query并查询知识库后生成，或通过 SMT 模型生成，或通过 encoder-decoder 框架生成，有些 QA 场景可能还需要逻辑推理才能生成回复
+ 检索式问答系统典型场景
	+ 1）候选集先离线建好索引；
	+ 2）在线服务收到 query 后，初步召回一批候选回复；
	+ 3）matching 和 ranking 模型对候选列表做 rerank 并返回 top K。
+ NN 实现语义匹配的典型工作

##### 1. Po-Sen Huang, et al., 2013, Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
+ 相似性问题： 计算 Quary 和 Doc 的相似性
+ **这篇博客讲的很好： https://cloud.tencent.com/developer/article/1005600**
+ From UIUC 和 Microsoft Research
+ 针对搜索引擎 query/document 之间的语义匹配问题 ，提出了基于 MLP 对 query 和 document 做深度语义表示的模型（Deep Structured SemanticModels, DSSM）
+ structure : 
	![](https://pic1.zhimg.com/80/v2-0187cc3483ec2a2f88453576eef61cc5_hd.jpg)
+ Step
	+ 先把 query 和 document 转换成 BOW 向量形式
	+ 然后通过 word hashing 变换做降维得到相对低维的向量（备注：除了降维，word hashing 还可以很大程度上解决单词形态和 OOV 对匹配效果的影响）
		+ word hashing
			+ http://blog.csdn.net/washiwxm/article/details/19838595
			+ 举个例子，假设用 letter-trigams 来切分单词（3 个字母为一组，#表示开始和结束符），boy 这个单词会被切为 #-b-o, b-o-y, o-y-#
			+ 这样做的好处有两个：首先是压缩空间，50 万个词的 one-hot 向量空间可以通过 letter-trigram 压缩为一个 3 万维(27*27*27)的向量空间。其次是增强范化能力，三个字母的表达往往能代表英文中的前缀和后缀，而前缀后缀往往具有通用的语义。
			+ 选择三字母的原因
	+ 喂给 MLP 网络，输出层对应的低维向量就是 query 和 document 的语义向量（假定为 Q 和 D）
	+ 计算(D, Q)的 cosinesimilarity 后
	+ 用 softmax 做归一化得到的概率值是整个模型的最终输出，该值作为监督信号进行有监督训练

##### 2. Yelong Shen, et al, 2014, A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval
+ 这篇文章出自 Microsoft Research，是对上述 DSSM 模型的改进工作
+ 在 DSSM 模型中，输入层是文本的 bag-of-words 向量，**丢失词序特征，无法捕捉前后词的上下文信息**
+ 基于此，本文提出一种基于卷积的隐语义模型（convolutional latent semantic model, CLSM）
+ Structure :
	![](https://pic1.zhimg.com/80/v2-29ffcff7590aea70e85df0deb3d71abe_hd.jpg)
+ Step:
	+ 先用滑窗构造出 query 或 document 的一系列 n-gram terms（图中是 trigram），
	+ 然后通过 word hashing 变换将 word trigram terms 表示成对应的 letter-trigram 向量形式（主要目的是降维）
	+ 接着对每个 letter-trigram 向量做卷积，由此得到「Word-n-gram-Level Contextual Features」
	+ 接着借助 max pooling 层得到「Sentence-Level Semantic Features」
	+ 最后对 max pooling 的输出做 tanh 变换，得到一个固定维度的向量作为文本的隐语义向量
	+ Query 和 document 借助 CLSM 模型得到各自的语义向量后，构造损失函数做监督训练
	+ 训练样本同样是通过挖掘搜索点击日志来生成
+ Experiment:
	+ BM25、PLSA、LDA、DSSM
	+ NDCG@N 指标表明，CLSM 模型在语义匹配上达到了新的 SOTA 水平
	+ 文中的实验和结果分析详细且清晰，很赞的工作

##### 3. Zhengdong Lu & Hang Li, 2013, A Deep Architecture for Matching Short Texts
+ From : 这篇文章出自华为诺亚方舟实验室
+ Scenario : 针对短文本匹配问题
+ 提出一个被称为 DeepMatch 的神经网络语义匹配模型
+ 该模型的提出基于文本匹配过程的两个直觉：
	+ 1）Localness，也即，两个语义相关的文本应该存在词级别的共现模式（co-ouccurence pattern of words）；
	+ 2）Hierarchy，也即，共现模式可能在不同的词抽象层次中出现。

##### 4. Zongcheng Ji, et al., 2014, An Information Retrieval Approach to Short Text Conversation
+ From : 这篇文章出自华为诺亚方舟实验室
+ Scenario : 针对的问题是基于检索的短文本对话，但也可以看做是基于**检索的问答系统**
+ Step:
	+ 主要思路是，从不同角度构造 matching 特征，作为 ranking 模型的特征输入。
	+ 构造的特征包括：
		+ 1）Query-ResponseSimilarity；
		+ 2）Query-Post Similarity；
		+ 3）Query-Response Matching in Latent Space；
		+ 4）Translation-based Language Model；
		+ 5）Deep MatchingModel；
		+ 6）Topic-Word Model；
			+ 7）其它匹配特征。

##### 5. Baotian Hu, et al., 2015, Convolutional Neural Network Architectures for Matching Natural Language Sentences
+ From : 华为诺亚方舟实验室
+ 采用 CNN 模型来解决语义匹配问题，文中提出 2 种网络架构，分别为 ARC-I 和 ARC-II
+ ARC-I
	+ Structure
	![](https://pic1.zhimg.com/80/v2-bf0d6e2b0040fa995b1d3cadf3b8bb56_hd.jpg)
	+ Step :
		+ 上图所示的 ARC-I 比较直观，待匹配文本 X 和 Y 经过多次一维卷积和 MAX 池化，得到的固定维度向量被当做文本的隐语义向量，
		+ 这两个向量继续输入到符合 Siamese 网络架构的 MLP 层，最终得到文本的相似度分数。
		+ 需要说明的是，MAX POOLING 层在由同一个卷积核得到的 feature maps 之间进行两两 MAX 池化操作，起到进一步降维的作用。
		+ 作者认为 ARC-I 的监督信号在最后的输出层才出现，**在这之前，X 和 Y 的隐语义向量相互独立生成，可能会丢失语义相关信息，于是提出 ARC-II 架构**。
+ ARC-II
	+ (to be continued)

##### 6. Lei Yu, et al., 2014, Deep Learning for Answer Sentence Selection
+ From : University of Oxford 和 DeepMind
+ 提出基于 unigram 和 bigram 的语义匹配模型
+ Step :
	+ unigram :
		+ 其中，unigram 模型通过累加句中所有词（去掉停用词）的 word vector，
		+ 然后求均值得到句子的语义向量；
	+ bigram :
		+ bigram 模型则先构造句子的 word embedding 矩阵，
		+ 接着用 bigram 窗口对输入矩阵做 1D 卷积，
		+ 然后做 average 池化，
		+ 用 n 个 bigram 卷积核对输入矩阵分别做「1D 卷积+average 池化」后，会得到一个 n 维向量，作为文本的语义向量
	+ 对(question,answer)文本分别用上述 bigram 模型生成语义向量后，计算其语义相似度并用 sigmoid 变换成 0~1 的概率值作为最终的 matching score。该 score 可作为监督信号训练模型。
	+ Structure
	![](https://pic3.zhimg.com/80/v2-cd4c9f238689d0412754b3761b84a6af_hd.jpg)
+ 文中用 TREC QA 数据集测试了提出的 2 个模型，实验结果的 MAP 和 MRR 指标表明，unigram 和 bigram 模型都有不错的语义匹配效果，其中 bigram 模型要优于 unigram 模型。
+ 特别地，在语义向量基础上融入 idf-weighted word co-occurence count 特征后，语义匹配效果会得到明显提升。文中还将提出的 unigram 和 bigram 模型与几个已有模型进行了效果对比，结果表明在同样的数据集上，融入共现词特征的 bigram 模型达到了 SOTA 的效果。
##### 7. Aliaksei Severyn, et al., 2015, Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks
##### 8. Ryan Lowe, et al., 2016, The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems
+ From : McGill 和 Montreal 两所大学
+ Scenario : 针对基于检索的多轮对话问题，提出了 **dual-encoder** 模型对 context 和 response 进行语义表示，该思路也可用于检索式问答系统
+ Structure :
![](https://pic1.zhimg.com/80/v2-c4be342ff33fbefc8d0953a7d7bfd1ed_hd.jpg)
+ Step :
	+ 通过对偶的 RNN 模型分别把 context 和 response 编码成语义向量，
	+ 然后通过 M 矩阵变换计算语义相似度，
	+ 相似度得分作为监督信号在标注数据集上训练模型。
		+ 文中在 Ubuntu 对话语料库上的实验结果表明，dual-encoder 模型在捕捉文本语义相似度上的效果相当不错。

##### 从上面 8 篇论文可知，与关键词匹配（如 TF-IDF 和 BM25）和浅层语义匹配（如隐语义模型，词向量直接累加构造的句向量）相比，基于深度学习的文本语义匹配模型在问答系统的匹配效果上有明显提升。