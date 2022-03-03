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

## Distributed Training

