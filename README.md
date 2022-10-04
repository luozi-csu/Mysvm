# Mysvm

#### 介绍
机器学习课设

#### 使用说明
1. Visual Studio新建空项目，将数据集和main.cpp拷贝到项目根目录（注意两个文件在同一级目录中），在项目资源管理器中添加源文件main.cpp，添加资源文件agaricus-lepiota.data
2. 先调用InitData函数初始化数据集，第一个参数为数据集文件路径，第二和第三个参数为留出法比例（训练集大小比测试集大小）
3. 基学习器训练：调用SVM构造函数创建模型，第一个参数为训练集，第二个参数为正则化常数，默认使用留出法产生的测试集做测试，每过1个训练批次就会输出一次测试准确率，最终得到平均准确率
4. bagging集成学习：直接调用bagging函数，第一个参数为产生基学习器的个数，第二个参数为SVM模型的正则化常数，输出每个模型的平均准确率和集成学习后的准确率
