# Current Solution for Question Answering by Searching

## Reference
- [1] 基于知识的智能问答技术　冯岩松
- [2] 基于深度学习的阅读理解　冯岩松
- [6] FastQA 论文笔记
	- http://www.shuang0420.com/2018/05/13/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%20-%20Making%20Neural%20QA%20as%20Simple%20as%20Possible%20but%20not%20Simpler/
- [7] 《Fast and Accurate Reading Comprehension by Combining Self-attention and Convolution》阅读笔记
	- https://zhuanlan.zhihu.com/p/35186608


# Solution 1 知识库相关方案 [1]
+ 知识库 :　Freebase, DBpedia, Wikipedia, yago, satori
+ 常见数据集
	+ ATIS
	+ GeoQuery
	+ QALD Series
	+ Free 917
	+ Web Questions
	+ Web QuestionsSP
	+ SimpleQuestions
	+ WikiMovies
	+ TriviaQA
+ 存在的问题
	+ 面向特定领域
	+ 知识库规模有限
	+ 依赖手工设计的模板或规则来解析问题
	+ 难以应付开放领域
+ 技术挑战
	+ 如何恰当的表示问题的语义
		+ 丰富的提问方式,复杂的提问意图 ----> 语义表示,语义分析
	+ 如何利用(大规模)(开放域)知识库元素来表示问题的含义  
		+ 大规模,开放域 ----> 知识库映射 ----> 实体链接,关系抽取
	+ 需要什么样的知识来解答
		+ 知识的多样化 ---->知识融合
	+ 传统方法
		+ 优势 : 语义表达能力强
		+ 缺点 : 数据规模膨胀（优质训练数据有限，噪声数据居多，对稀有类型和新拓展类型数据利用有限）
	+ DL
		+ 优点 : 可以处理大规模知识及语料,**有效利用噪声**
		+ 缺点 : 语义表达能力，常识，推理能力较弱

## Problem 1 根据知识库输出 实体答案[1]
+ 输入 : 自然语言语句
+ 资源 : 结构化知识库, 文本知识, 表格, 结构化/半结构化记录
+ 输出 : 答案
 
### Solution 1.1 :语义分析类方案［１］
+ Nautural Language Question  --> **Senmantic Parser** -->   Meaning Representation   --> Map with KB --> Structured Query --> Query over KB --> Answer
	+ 对输入文本进行语义解析,并在KB中进行结构化查找来确定最终答案
+ 语义分析
	+ 利用形式化方法表达问题语义
+ 语义表示
	+ $\lambda$-Calculus
		+ 找到所有实体, 在遍历所有实体属性, 得到答案
		+ 如世界最大的国家是哪个? 可以先找到所有的国家, 然后在找到每个国家的面积, 在找到面积最大的那个国家
	+ $\lambda$-DCSs
		+ Lambda Dependency-based Compositional Semantics 
	+ CCG
		+ Combinatory Categorial Grammer
		+ WebCorpus 经过CCG 处理 构造组合语法
	+ Simple Query Graph (注重实体抽取与关系抽取)
		+ 事实类问题只涉及实体属性及关系
		+ 疑问词,实体及关系组成查询图
		+ 实体与关系抽取联合优化
	+ Query Graph
		+ 直接有效利用疑问词,实体,关系及额外节点组成查询图
	+ Phrase Dependency Graph (短语依存图)
		+ 独立于某个知识库的语义结构表示
			+ 短语检测及与知识库无关的短语分类
				+ Variable, Category, Entity and Relation
			+ 短语依存图解析
		+ 灵活映射到具体的知识库中
            + 基于状态转移系统的解析器
            + 联合消解
            	+ 两个子任务之间存在错误传递
            + Grounding (从短语依存图到可执行的结构化查询)
            	+ 实体链接
            		+ 核心实体决定成败
            		+ 缺乏足够上下文, 命名实体边界模糊
            	+ 概念匹配
            	+ 关系抽取/匹配
            		+ 上下文有限,表达方式灵活多样,与知识库表达不匹配,候选关系众多,错误传递
            		+ 复杂关系,如N-ray关系,事件,或者Freebase中的CVT节点
            		+ 方法:关系抽取,通过实体/类别猜猜,联合消解
            	+ 各个元素与知识库的映射是独立进行的
            	+ 最终结果仍可能存在错误积累

### Solution 1.2 : 信息抽取类方案［１］
+ Nautural Language Question --> Anchor --> Topic Entity --> Retrieve KB Graph --> Candidate Answer --> Ranking --> Answer
	+ 找到最主要的实体(Topic Entity), 并在KB中查找此实体,获得一系列候选答案,并对候选答案排序来确定最终答案
+ IEQA
	+ Freebase Topic Graph
+ 优点
	+ 架构灵活,实用
	+ 易于融合多种线索
	+ 容易与其他方法/框架结合
	+ 适用于多种类型的资源
+ 缺点
	+ 依赖特征工程
	+ 易受错误传递影响
	+ 不擅长语义组合
	+ 难以处理推理问题 

#### Summary : 知识问答中的关系抽取 
+ 挑战
 	+ 大规模数据集中通常包含上万种关系
 	+ 通常只能收集到噪声数据集
+ 任务定义
 	+ 已知一句子及句子中的一对实体,求这对实体间的语义关系
+ 传统方法
	+ 依赖特征工程抽取特征
		+ 词法特征, 如Pos, WOrdNet, FrameNet特征等
		+ 句法特征, 如句法结构特征等
	+ 深度学习的方法希望摆脱特征工程
		+ 各种神经网络结构
		+ 利用词向量及网络结构抽取词法特征和句子级特征
+ 神经网络模型
	+ pass 
		+ 相关论文没找到
+ 依存关系结构
	+ 两个名词短语在依存结构上的最短路径
		+ 提供丰富的语义信息
		+ 可减轻复杂句法结构带来的困扰
		+ 具有很好的抗噪声能力
+ Multi-Channel Convolutional Neural Networks
	+ pass
+ 关系抽取的测试
	+ WebQuestions
	+ SemEval-2010 Task 8
		+ http://www.aclweb.org/anthology/S10-1006
			+ SemEval 国际语义测评

### Solution 1.3 : 基于深度学习的解决方案［1］
+ 对现有模块的改进
	+ 语义分析类
		+ STAGG
			+ Staged Query Graph Generation
				+ 通过搜索，逐步构建查询图（Query Graph）
    + 信息抽取(候选评分)
        + Linking Topic Entity
        	+ 依靠高可靠性的实体链接工具
        	+ 保留Top 10 作为后续模块的候选输入
        + Identifying Core Inferential Chain
        	+ 基于CNN的关系抽取
        + Argument Constraints
        	+ 逐步加入限制节点
        	+ 限制节点候选:实体,约束性关键词
        		+ 依赖规则产生限制节点
        + Learning
        	+ 以排序为目标
        		+ Log Linear Model
        		+ 正确的Query Graph 得分更高
        		+ 可利用中间步骤获得大量训练数据
        	+ 汇总全流程的各种特征,中间得分 
        		+ 实体链接得分
        		+ 关系抽取得分
        		+ 约束关键词匹配得分

+ Neural End-to-End 框架 (多数遵循 **信息抽取** 类框架)
	+ Embedding Everything 
		+ End-to-End (主要基于信息抽取类)
			+ 输入文本直接embedding
		+ Large-scale Simple Question Answering with Memory Networks
 			+ https://arxiv.org/pdf/1506.02075v1.pdf
		+ Simple Matching
			 + 只处理单关系的简单问句
			 + 典型的信息抽取框架
				+ 候选生成 : Topic Entity 
				+ 排序选择最优
				+ 多任务优化:文本复述(???)
				+ 候选答案的上下文
				+ 候选答案到Topic Entity的上下文
		+ Multi-Column CNNs
			+ 基于典型的信息抽取框架
			+ 利用 MCCNN 抽象问题的不同侧面
			+ 利用知识库从不同的侧面刻画候选答案
			+ (有个图不错,需要加上)
 
	+ Attention && Global Knowledge
		+ 之前方法的问题
			+ 问句语义表示过于简单
			+ 对实体名称等的训练数据不足
		+ 改进
			+ Cross-Attention : 刻画问句表达与答案间的关联
				+ 【论文笔记】An End-to-End Model for QA over KBs with Cross-Attention Combining Global Knowledge
					 + https://blog.csdn.net/LAW_130625/article/details/78484866
                        + TransE : 多任务学习
		+ (有两张图需要添加)
		+ (CrossAttention 和 TransE 需要仔细看)
    	
    + Memory Networks
    	+ 简介
    		+ QA 方向较早的尝试
    		+ 面向简单问题:只需一条答案
    		+ 记忆单元
    			+ 每个单元储存一个知识三元组
    			+ 词袋模式
    		+ 自然语言问题
    			+ 词袋模式
    			+ 问题与存储单元的简单计算(借助权值矩阵)
			+ 有效性验证
        + 方法
        	+ 语义分析类 :　Key-Value Memory Networks
        		+ 记忆单元为 Key-Value 形式, 如<主体-关系, 客体>
        		+ 访问时, Query 与 存储单元的Key计算相似度
        		+ 检索得到的对应Value 用来更新 Query
        		+ 支持多次访问 Memory --> 潜在支持浅层推理:语义组合
        	+ 语义分析类型：Neural Symbolic Machines
        		+ 学习查询步骤(符号系统)(不懂这个符号系统是啥意思)
        			+ 期望 : 问题 -> 查询 -> 答案 (学习如何查询)
        			+ Seq2Seq with Key-Variable Memory
        				+  由问题得到查询命令
        			+ 需要支持生成 查询语句,函数
        			+ 弱监督框架存在问题 : Non-Differentiable
        				+ 加图
        			+ 训练策略
        				+ reinforce
        				+ augmented reiforce 
        				+ 加图

## Problem 2 根据知识库生成 自然回答[1]
+ 输入:事实类自然语言问题
+ 输出:生成自然语言回答
	+ 回答正确
        + 表述自然
        + 融入到对话等自然应用场景中去

### Solution 2.1  GenQA [1]
+ Encoder-Decoder 框架
	+ 读入并表示问题
	+ 查询知识库,获取三元组
	+ 生成自然语言回复
		+ example : "how tall is Yao Ming?" -> "He is 2.26m and visiable from space"
		+ 加图
    + GenQA: Automated Addition of Architectural Quality Attribute Support for Java Software
    + http://selab.csuohio.edu/~nsridhar/research/Papers/PDF/genqa.pdf

### Solution 2.2 COREQA [1]
+ Encoder-Decoder 框架
+ 通过实体查询三元组
+ 问题与知识编码
+ 解码生成答案
+ 检索,拷贝机制与状态更新
+ 加图

## Summary of Solution 1 : 基于实体关系的问答技术[1]
+ 基于实体关系的问答技术
    + 结构化知识库不足够
    	+ example
    		+ what mountain is the **highest** in north america ?
    		+ who did shaq **first** play for ?
    		+ who is shaq's farther ?
    	+ 实体与关系的联合消解
    		+ Joint Entity Linking and Relation Extraction(Randking problem)
    	+ 其他文本的作用：利用维基正文清洗候选答案
    		+ Refinement model
    		+ 有个 奥尼尔在最开始在哪支球队打球的例子
    + Hybrid-QA:基于混合资源的知识库问
    	+ 局限
    		+ 知识库回答的性能滞后
    	+ 解决方案:
    		+ 使文本更好的融入解答过程
    	+框架
        	+ 利用依存结构来分解
        + 知识库实体链接
            + Entity Linking
                + Freebase  - SMART entity linking tool
                + DBpedia - DBpedia lookup tool
        + 知识库关系抽取
            + KB-based Relation Extraction
                + Multi Convolution Neural Network Based Model
                    + Freebase - WebQuestions
                    + DBpedia - PATTY
        + 文本关系抽取
            + Open Relation Extraction

# Solution 2 : 阅读理解 Reanding Comprehension  [2]
+ 背景
	+ 核心
	+ 形式
		+ 文档形式
		+ 问题形式
	+ 阅读理解 VS 传统问答
		+ 传统
			+ 各种知识资源
			+ 问题类型有限
			+ 答案类型有限
			+ 侧重对问题语义的准确表达
		+ 阅读理解
            + 答案来源-固定文档  
            + 问题形式多样
            + 答案类型多样
            + 根据上下文进行推理
	+ 挑战
		+ 需要多种形式的推理能力
		+ 需要篇章知识
		+ 需要常识知识
		+ 需要综合运用
	+ 典型任务
		+ 给定指定文档
		+ 问题形式
			+ 选择题
			+ 填空题
			+ 完形填空
		+ 典型数据集
			+ MCTest
				+ 儿童读物,选择题
			+ bAbi
				+ 简短故事,问答
			+ CNN/DailyMail
				+ 新闻,完形
			+ CBTest
				+ 儿童读物,完形
			+ Chinese RC
				+ 新闻/儿童读物,完形,iFlytex/HIT
			+ SQuAD
				+ 维基百科,问答
				+ 10W 问题-答案对,536篇文章
			+ TriviaQA
				+ 维基百科/网页,问答
			+ Science Exams
				+ 美国小学科学试题,选择题
+ 传统方法
	+ 两步架构
		+ 片段检索
			+ P(片段|问题,文档)
		+ 答案生成
			+ P(答案|问题,片段)
		+ 最终
			+ P(答案|问题,片段) * P(片段|问题,文档)
		+ 特征
			+ 挖掘隐形文本蕴含的结构,如,词级别的对应特征
				+ Sachan, ACL 2015
	+ 问题
		+ 与问答任务的困难类似
			+ 词汇,表达,常识(不理解)
		+ 对篇章理解的建模能力有限
		+ 对深层次的推理需求无能为力
			+ bAbi 中有20中任务类型
		+ 外部工具,资源带来的错误传递与积累
+ 深度学习方法
	+ LSTM
		+ Deep LSTM Reader
			+ 对距离较远的关键词缺乏足够的关联建模
	+ Attention Mechanism
		+ Attentive Reader
			+ Bi-LSTM + Attention
			+ 找到最有支持力度的句子
		+ Impatient Reader
			+ LSTM + Attentions
			+ 在处理问题的单词时, 通过注意力机制令模型能够重新阅读理解文档句子
			+ 逐步处理问题,反复阅读句子,产生更好的文档表示
	+ Memory Network
		+ I(Input feature map):将输入转化为**内部特征**表示
		+ G(Generalization):根据输入**更新**当前Memory
		+ O(Oputput feature map):根据**输入**和**当前Memory状态**，生成**输出向量**
		+ R(Response):根据**输出向量**，产生答案
		+ 改进
			+ 自适应的记忆单元
			+ 记忆单元使用N-Gram
			+ 函数匹配非线性化
			+ 每一步都需要完全监督, 如何处理多步推理
		+ End2End MN
			+ 记忆单元/bAbi
	+ Attention over Attention
	+ Match LSTM
	+ Bi-directional Attention Flow(BiDAF)
	+ Doc Retriever-Doc Reader
		+ Open-domain QA
			+ SQuAD, TREC, Web Question, WikiMovies
    	+ Mnemonic Reader
    	+ R-Net
    	+ Self-Matching Network   
	+ Hierarchical CNN
	+ Pointer Network

+ 比较
	 + 基于传统特征
	 	+ 难以适应大规模,开放域的问题
	 	+ 针对不同类型的阅读理解任务需要重新设计
	 + 基于深度学习
		+ 受益于词向量的分布式表示和各种结构
		+ 专注于模型结构
		+ 不同形式的推理建模还有很大的提升空间