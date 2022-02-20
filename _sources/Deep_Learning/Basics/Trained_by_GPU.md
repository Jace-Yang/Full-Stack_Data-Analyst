# GPU训练

### Single Node, Single GPU Training

Training throughput depends on:
- Neural network model (activations, parameters, compute operations)
- Batch size
- Compute hardware: GPU type (e.g., Nvidia M60, K80, P100, V100)
- Floating point precision (FP32 vs FP16)
    - 换了FP16之后相当于working with reduced precision arithmetic｜Using FP16 can reduce training times and enable larger batch sizes/models without significantly impacting the accuracy of the trained model
    
训练时间：Training time with single GPU very large: 6 days with Places dataset (2.5M images) using Alexnet on a single K40.

提高performance的办法一般是Increasing batch size increases throughput
- 然而Batch size is restricted by GPU memory！
- GPU的限制下⇒Small batch size $=>$ noisier approximation of the gradient $\Rightarrow$ lower learning rate $=>$ slower convergence



### DL Scaling with GPU

#### 硬件选择：Commonly used GPU Accelerators in Deep Learning

<img src="https://cdn.mathpix.com/snip/images/EgvWEJqCtlVEjyoLDQ1kqDWtRSVN5q_ZW9jcmnqsJFM.original.fullsize.png" />


- 两个K80就可以实现6个TFLOPS

#### 对比单GPU的提升

<img src="https://cdn.mathpix.com/snip/images/ym85dUm7bzwccOmHZq03IOOmqBdaK34LrMLl783Nyws.original.fullsize.png" />

- 不同模型的speedup不一样，因为different networks have different structure, they have different amount of floating point operations requirement, they have different memory requirement

<img src="https://cdn.mathpix.com/snip/images/F8v3Uz9CFYVg0XIHolZnrAKvV4u5H-AE1mapfdeJrMQ.original.fullsize.png" />
- 从batch size提高后的improve来看也是一样的model dependent！

### Single Node, Multi-GPU Training

最重要的是communicate的cost！让Scaling not linear的各种影响因素——Synchronization
- Communication libraries (e.g., NCCL) and supported communication algorithms/collectives (broadcast, all-reduce, gather)
    - NCCL ("Nickel") is library of accelerated collectives that is easily integrated and topology-aware so as to improve the scalability of multi-GPU applications
- Communication link bandwidth: PCle/QPI or NVlink
- Communication algorithms depend on the communication topology (ring, hub-spoke, fully connected) between the GPUs.
- Most collectives amenable to bandwidth-optimal implementation on rings, and many topologies can be interpreted as one or more rings [P. Patarasuk and X. Yuan]

<img src="https://cdn.mathpix.com/snip/images/XWDeJcrg4WsPOm1CDXkWF005zbjCK7KoXanRu0yo630.original.fullsize.png" />

- V100优势在于GPU和GPU之间的connectivity很好

没有NVLINK的时候：

<img src="https://cdn.mathpix.com/snip/images/pwqaKYxymHmAt-5HJ7HR37vmC5lgVfR8ITd73dYZVjE.original.fullsize.png" />

有NVLINK的时候：

<img src="https://cdn.mathpix.com/snip/images/bFRPoR_Y-bsmIVYEzzLtJzSnchkKboILT1Zh3O7HCus.original.fullsize.png" />

<img src="https://cdn.mathpix.com/snip/images/EUtejV4EyMXJtlUlGNuAA6XayYDeicPhJlmuCSw_oH4.original.fullsize.png" />

<img src="https://cdn.mathpix.com/snip/images/6UpUK2ib1ACVqIj-N7a42p-Gs-V2xUZrTcFTkAWHbD8.original.fullsize.png" />

### Distributed Training 

> multiple nodes in multiple machines (GPUs): parallelism by distributing your job across multiple machines

#### Partition and synchronize:
- Type of Parallelism: Model, Data, Hybrid
    - Data Parallelism: partition the data set across multiple machines and then each one of them can do the training using that dataset (但是有整个model)
    - Model Parallelism: train portions of a big model on different machines (但用整个dataset)
    - Hybrid: partition the model and also partition the data.

#### Aggregation
- 目的
    - **Average model得不到我们想要的global的model**：partition之后即使每个模型看到的local sub-dataset的分布一样，但因为features的区别还是会train出不一样的parameters. 因为每个local model不会向总体generalize，所以Averaging them之后也不会generalized 
    - 因此，我们想要的是train一些iterations之后就做synchronize, 假设不分batch，做data parallelism的时候：
        <center><img src="../../images/DL_GPU_1.png" width="45%"/></center>

        - 开始先initialize with same weight
        - forward pas get the loss
        - backward get the gradient
        - sum the gradient （会$\iff$用总loss算的gradient）
        - 然后让每个submodel从这套parameter再继续迭代
    
        如果分了batch size n：
        - 那就总共就pass $n \times m$个samples
        - 得到$n \times m$个gradients
        - 再计算$\frac{\partial}{\partial w} \sum_{i=1}^{m} \sum_{j=1}^{n}Loss_{i, j}$

- Type of Aggregation: Centralized, decentralized
    - <mark style="background-color:#e1e1e1;">Centralized aggregation: parameter server</mark>——就是把gradient传送到centralized server，center的只maintain model parameter不做training
    - <mark style="background-color:#e1e1e1;">Decentralized aggregation: P2P, all reduce</mark>——直接让workers之间交流实现aggregation
     <center><img src="../../images/DL_GPU_2.png" width="45%"/></center>
    
    
#### Performance metric: Scaling efficiency：
$$\frac{\text{run time of one iteration {\bf on a single GPU}}}{\text{the run time of one iteration when {\bf distributed over $n$ GPUs}}}$$ 
- batch size一样的话：对于n个GPU来说 是$n \times$ Single GPU batch size，但是是parallel进行，所以每个GPU还是处理一样的数据！因此时间in 2 cases应该是一样的 $\Rightarrow \text{scaling efficiency} = 1$
- 然而，GPU中需要communication的时间——Sychronize to get an aggregated gradient，所以会有更长的时间！ $\Rightarrow \text{scaling efficiency} < 1$


#### Parallelism
Parallel execution of a training job on different compute units through 
- scale-up (single node, multiple and faster GPUs)：让one machine的比如4个GPU 连接很快，实现powerful
- scale-out (multiple nodes distributed training) ：用多个machine（nodes），可能每个都不是很贵但可以堆，不过这样network会成为bottleneck

方式：
- Enables working with large models by partitioning model across learners
- Enables efficient training with large datasets using large "effective" batch sizes (batch split across learners)
- Speeds up computation

##### Model Parallelism

Splitting the model across multiple learners，比如下面这个5层神经网络分给4个learner（learner之间连接加粗）

 <center><img src="../../images/DL_GPU_3.png" width="45%"/></center>

- 做Partition的criteria
    - minimize这些bold edges
    - 也就是keep the nodes densely connect to each other in a same machine ⇒ exploit machine 的local compute power

Performance benefits depend on
- Connectivity structure
- Compute demand of operations

Heavy compute and local connectivity -benefit most
- Each machine handles a subset of computation
- Low network traffic

##### Data Parallelism

 <center><img src="../../images/DL_GPU_4.png" width="45%"/></center>

 - Model is replicated on different learners
- Data is sharded and each learner work on a different partition
- Helps in efficient training with large amount of data
- Parameters (weights, biases, gradients) from different replicas need to be synchronized

###### Parameter server (PS) based Synchronization

Synchronous SGD：
>Synchronous: at any instance of time the model at each of the workers is same (always working on the same model)

 <center><img src="../../images/DL_GPU_5.png" width="45%"/></center>

1. Each learner executes the entire model
1. After each mini-batch processing a learner calculates the gradient and sends it to the parameter server
1. The parameter server calculates new value of weights and sends them to the model replicas

`模型太大的问题`：partition into model shards，这样一个机器只用responsible for maintaining some parameters

`沟通问题`：当模型数量很大的时候，bottleneck可能会变成bandwidth

`内存问题`：模型很大的时候没法fit in single machine
- 注意 比如每个模型10000个weight，100个模型，那parameter serve需要先接受100*10000，再average成10000个weight

`Straggler problem问题`：PS needs to wait for updated gradients from all the learners before calculating the model parameters在center收集gradient的时候learner只能idle 等最慢的worker好了然后得到new parameter完成一次iteration

- 具体解释：

    - Even though size of mini-batch processed by each learner is same, updates from different learners may be available at different times at the PS
        - Randomness in compute time at learners
        - Randomness in communication time between learners and Parameter Server
    - Waiting for slow and straggling learners diminishes the speed-up offered by parallelizing the training


- 解决方案1——Synchronous SGD Variants
 
    <center><img src="../../images/DL_GPU_6.png" width="100%"/></center>
    
    > P: total number of learners<br>
    K: number of learners/mini-batches the PS waits for before updating parameters<br>
    Lightly shaded arrows indicate straggling gradient computations that are canceled.

    - K-sync SGD: PS waits for gradients from K learners before updating parameters; the remaining learners are canceled
        - When K = P, K-sync SGD is same as Fully Sync-SGD
    - K-batch sync: PS waits for gradients from K mini-batches before updating parameters; the remaining (unfinished) learners are canceled
        - 特点
            - Irrespective of which learner the gradients come from
            - Wherever any learner finishes, it pushes its gradient to the PS, fetches current parameter at PS and starts computing gradient on the next mini-batch based on the same local value of the parameters
        - Runtime per iteration reduces with K-batch sync; error convergence is same as K-sync

- 解决方案2——Asynchronous SGD and variants

    <center><img src="../../images/DL_GPU_7.png" width="100%"/></center>

    - Async-SGD: do not need to guarantee that all the workers locally are working with the same model
        - 跟1-sync-SGD的异同：同样只用等一个model的结果，但不会把这个结果update到所有的learner而kill慢learner的结果，而是
    - K-async SGD: PS waits for gradients from K learners before updating parameters but the remaining learners are not canceled; each learner may be giving a gradient calculated at stale version of the parameters
        - When $K=1$, K-async SGD is same as Async-SGD
        - When $K=P$, K-async SGD is same as Fully Sync-SGD
    - K-batch async: PS waits for gradients from K mini-batches before updating parameters; the remaining learners are not canceled
    - Wherever any learner finishes, it pushes its gradient to the PS, fetches current parameter at PS and starts computing gradient on the next mini-batch based on the current value of the PS
    - Runtime per iteration reduces with K-batch async; error convergence is same as K-async

    单位时间更多update会让model收敛更快，但也会遇到新的`Stale Gradients`的问题

    <center><img src="../../images/DL_GPU_8.png" width="80%"/></center>

    - PS updates happen without waiting for all learners ⇒ Weights that a learner uses to evaluate gradients may be old values of the parameters at PS
        - Parameter server asynchronously updates weights
        - By the time learner gets back to the PS to submit gradient, the weights may have already been updated at the PS (by other learners)
        - Gradients returned by this learner are stale (i.e., were evaluated at an older version of the model)
    - Stale gradients can make SGD unstable, slowdown convergence, cause sub-optimal convergence (compared to Sync-SGD)

