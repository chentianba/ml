## 神经网络
### 相关文章
* [详细解读神经网络十大误解，再也不会弄错它的工作原理](http://blog.sina.com.cn/s/blog_806ac7d70102yku4.html)  

### 神经网咯设计要点
1. 隐藏层数量  
![隐藏层数量设置](https://github.com/chentianba/ml/blob/master/neural_network/hidden_layer_setting1.jpg)  
2. 使用梯度下降更新参数  
由于**神经网络的优化不是凸优化问题**，因此一般不使用设定阈值来使其停止更新，一般按照迭代次数停止更新，也就是说把迭代次数跑完参数才会停止更新。  
