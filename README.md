# HaarvsLogistic

这是一个作业项目，为了完成Haar和Logistic regression的人脸识别对比。

## 关于数据集
因为测试数据集只有Positive的数据，没有Negative的数据，所以基于给定数据集FPR无法算出，找了20个Negative的数据Sample。
存在Negative数据样本不足的问题，但用于本次作业的学习和研究是够了。

## 关于模型
1.Haar模型使用了Opencv的实现
2.Logistic regressio由于没找到合适的Pretrained的模型实现(大部分使用了Logistic Regression方式的模型都为Deep Neuron network),深度网络模型超出了本次作业的范围，因此没有使用。
  本次作业中使用老师给的数据集做为positive，从网上随机抓了一些图做为Negative 训练了一个Logistic Regression的模型。
  该模型的特征提取采用了lbp pattern。这个训练采用的数据集比较片面（特别是Negative数据集严重不足），因此该模型并不具备实际使用价值，仅可用于本次作业学习使用。
  有时间的话可以找更多的训练数据集，ROC绘制使用的数据集最好与训练用的数据集分开，能得到更客观的实验数据。

## 关于ROC的绘制
- ROC数据的运算采用了sklearn.metrics.roc_curve
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html 
- 绘制采用了 matplotlib.pyplot
- Haar模型的绘制
```python Haar.py ```
- logistic regression 模型的绘制
```python trainlogisticmodel.py```
```python logistic_regression.py```

## 一些有用的资源
- https://www.evidentlyai.com/classification-metrics/explain-roc-curve#roc-curve-in-python
- https://medium.com/trueface-ai/what-are-roc-curves-and-how-to-interpret-them-d53b09c06b81
