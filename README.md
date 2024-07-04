# 项目名称 Cifar-10图片分类问题

注意：代码中Train和Commitresult两处在本地运行的时候要用本地数据集保存的路径
Train中的
- data_dir = 'F:/PycharmProject/pythonProject6/train'
- csv_file = 'F:/PycharmProject/pythonProject6/trainLabels.csv'
改成自己的数据集、标签所在位置即可
在Commitresult也是如此即可

库依赖：
- torch
- os
- torchvision

解释器：
- python 3.11

项目结构说明：
- model为本项目的模型文件，是基于resnet101的修改
- Train为本项目的训练文件
- Commitresult将为本项目生成submission
  
运行方式：
- 直接点击Train即可进行训练
- 点击Commitresult即可生成submission提交文件用来提交Kaggle
  
准确率：
- 本项目提交kaggle准确率有95.7%

Kaggle比赛地址：
-https://www.kaggle.com/competitions/cifar-10/submissions


权重文件地址：
- 百度网盘链接：https://pan.baidu.com/s/1QyJTUsI4KOnAf2X7SSeadw 
提取码：pz88

特别注意：权重文件和代码文件将打包发送至邮箱


