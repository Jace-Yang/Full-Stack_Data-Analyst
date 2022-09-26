# 基础

这一部分cover一些深度学习的基础概念！



# Lecture 1 - 基础





<img src="../../images/image-20220919151957876.png" alt="image-20220919151957876" style="width:50%;" />



- LR无法提取features，尤其是pixels类的，所以需要DL进行representation layers来做特征提取
- 与其用手动的特征 找猫的眼睛 e.g，可以让Neural Network automatically learn useful representation of data





### Early Stopping

### Dropout

![img](../../images/(null)-20220726112646162.(null))

- 定义：Dropout is a regularization technique to deal with overfitting problem and improve generalization

  - dropout的含义：node is not used in this in the next round of training and and all its connections will be not be updated 
  - dropout的方式：randomly dropout probability for each hidden layers

- Prevents co-adaptation

   of activation units

  - Each hidden unit learns certain features **independently - the** features that each neuron is learning our robust features they are not dependent upon the presence or not presence of the subsequence of the previous layers!
  - 这样各个hidden unit学习的是different features

- Probabilistically drop input features or activation units in hidden layers

  - **Layer dependent** dropout probability  ( ~0.2 for input, ~0.5 for hidden)
  - 为什么要input小：you do not want to use a high value of report for the input 因为 most the input features will be gone right in the training而我们希望保留！但也可以允许一定的dropout

- Test的时候怎么处理dropout：apply dropout probability！这样test的时候其实layer都在





# References

- Columbia Applied Deep Learning 2022 Fall