# Distill Bert

## 前面的研究

有人用Bi-LSTM来实现作用在Fine-tuned的Bert上面

<img src="https://cdn.mathpix.com/snip/images/mi7BxT2oqEgFG21GzX9AMPrMNuUMbjUmCsrb1p9eilo.original.fullsize.png" />



## Loss

The final training objective:  a linear combination of the distillation loss $L_{c e}$, the masked language modeling loss $L_{m l m}$, and  a cosine embedding loss $L_{\cos }$: Loss= 5.0Lce+2.0 Lmlm+1.0* Lcos,



-  Lce: soft_label的KL散度: The student is trained with a distillation loss over the soft target probabilities of the teacher: $L_{c e}=\sum_{i} t_{i} * \log \left(s_{i}\right)$ where $t_{i}$ (resp. $s_{i}$ ) is a probability estimated by the teacher (resp. the student). 
   
    - This objective results in a rich training signal by leveraging the full teacher distribution. Following Hinton et al. [2015] we used a softmax-temperature: $p_{i}=\frac{\exp \left(z_{i} / T\right)}{\sum_{j} \exp \left(z_{j} / T\right)}$ where $T$ controls the smoothness of the output distribution and $z_{i}$ is the model score for the class $i$. 
    
    The same temperature $T$ is applied to the student and the teacher at training time, while at inference, $T$ is set to 1 to recover a standard softmax.

- $L_{m l m}$: hard_label的交叉熵

- 余弦相似度cosine embedding loss $L_{\cos }$: cosine similarity loss of hidden layer embedding between student and teacher

    - tend to align the directions of the student and teacher hidden states vectors.

### 第一和第三个是teacher model transfer knowledge to the student model的地方
