# Demo of GPU Multi Card Usage

# Parallelism
## Data Parallelism
+ 每个GPU上面跑一个模型，模型与模型之间结构参数相同, 只是训练的数据不一样
+ 每个模型通过最后的loss计算得到梯度之后, 再把梯度传到一个parameter server（PS）上进行参数平均average gradient
+ 然后再根据average gradient更新模型的参数
+ 同步模式
    +等到所有的数据分片都完成了梯度计算并把梯度传到PS之后统一的更新每个模型的参数。优点是训练稳定，训练出来的模型得到精度比较高；缺点是训练的时间取决于分片中最慢的那个片，所以同步模式适用于GPU之间性能差异不大情况下
+ 异步模式
## Model Parallelism
+ 当一个模型非常复杂，非常大，达到单机的内存根本没法容纳的时候，模型并行化就是一个好的选择。直观说就多多个GPU训练，每个GPU分别持有模型的一个片。它的优点很明显，大模型训练，缺点就是模型分片之间的通信和数据传输很耗时，所以不能简单说，模型并行就一定比数据并行要快

## Data and Model Parallelism

# 单机多卡

# 分布式多机多卡

# Tensorflow
+ https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py


