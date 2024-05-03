# HaarvsLogistic

这是一个作业项目，为了完成Haar和Logistic regression的人脸识别对比。

## 关于数据集
因为测试数据集只有Positive的数据，没有Negative的数据，所以基于给定数据集FPR无法算出，只能找了20个Negative的数据Sample。
可能存在Negative数据样本不足的问题。

## 关于模型
1.Haar模型使用了Opencv的实现
2.Logistic regression使用了

## 关于ROC的绘制
- ROC数据的运算采用了sklearn.metrics.roc_curve
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html 
- 绘制采用了 matplotlib.pyplot