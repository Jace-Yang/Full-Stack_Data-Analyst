# 神经网络压缩

模型压缩大体上可以分为5种

- **模型剪枝**：即移除对结果作用较小的组件，如减少 head 的数量和去除作用较少的层，共享参数等，ALBERT属于这种；
- **量化**：比如将 float32 降到 float8；
- **知识蒸馏**：将 teacher 的能力蒸馏到 student上，一般 student 会比 teacher 小。我们可以把一个大而深的网络蒸馏到一个小的网络，也可以把集成的网络蒸馏到一个小的网络上。
- **参数共享**：通过共享参数，达到减少网络参数的目的，如 ALBERT 共享了 Transformer 层；
- **参数矩阵近似**：通过矩阵的低秩分解或其他方法达到降低矩阵参数的目的；
- **NAS**: automated neural architecture search with reward function set as latency or model size

## 背景

Language model pre-training from large unlabeled data has become the new driving-power for models such as BERT, XLNet, and RoBerta. Built upon Transformer, BERT based models significantly improve the state of the art performance when fine-tuned on various Natural Language Processing (NLP) tasks.

Recently, many follow-up works push this line of research even further by increasing the model capacity to more than billions of parameters. Though these models achieve cutting-edge results on various NLP tasks, the resulting models have **high latency, and prohibitive memory footprint and power consumption for edge inference**. This, in turn, has limited the deployment of these models on embedded devices like cellphones or smart assistance, which now require cloud connectivity to function.

## Motivation


- 训练和部署之间存在着一定的不一致性: 
    - 在训练过程中, 我们需要使用复杂的模型, 大量的计算资源, 以便从非常大、高度冗余的数据集 中提取出信息。在实验中, 效果最好的模型往往规模很大, 甚至由多个模型集成得到。
    
        <center><img src="../../images/DL_NN_compress_3.png" width="55%"/></center>
        
        - 不断增长的参数量：比如BERT——BERT based models have a prohibitive memory footprint and latency. As a result, deploying BERT based models in resource constrained environments has become a challenging task.

        而大模型 不方便部署到服务中去, 常见的瓶颈如下:
        - 推断速度慢
        - 对部署资源要求高(内存, 显存等)
        - 在部署时, 我们对延迟以及计算资源都有着严格的限制。特别是edge devices像手机、智能手表
    
        这也就是模型压缩的动机：我们希望有一个规模较小的模型, 能达到和大模型一样或相当的结果。
- 这个问题因为用户隐私问题不能用上传到cloud做inference来解决

    A way to workaround: send data to the cloud, 但我们希望data privacy, don't want the data to leave uses' device. 之前会upload data to cloud，这样只用管理一部分的硬件不需要每个edge device都计算，但最近我们就看到migration back to device！这样不需要一直联网（提高了reliability，降低了latency），而且保护了用户隐私。

    <center><img src="../../images/DL_NN_compress_1.png" width="75%"/></center>

- 还有一个Background是边缘计算的能力增强了，有可能可以做到reduce latency without affecting model performance

    <center><img src="../../images/DL_NN_compress_2.png" width="75%"/></center>