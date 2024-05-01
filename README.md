# DATA620004：HW1
复旦大学 神经网络和深度学习作业1
仅使用numpy构建三层神经网络，在fashion-mnist数据集上实现图像分类

Homework_main.py中为完整流程，包括数据载入、模型训练、参数权重存储以及测试
Homework_search.py为参数网格搜索，主要对不同参数组合进行了模型训练效果评估。
weights_visual.py为训练权重参数可视化。

如何训练：
1、安装numpy、matplotlib、tqdm库
2、将数据集存放于/fashion-mnist/ 目录
3、根据需求调整Homework_main.py中网络setting和train_my_network（）的参数
4、运行Homework_main.py
训练完毕后权重_loss图像、验证集accuracy图像将存放到指定目录下；

关于测试：
训练参数保存后，会自动执行测试并返回测试集上的accuracy结果。也可以通过单独加载参数并调用测试函数进行测试。
可以使用weights_visual.py读取训练权重并实现权重可视化。

